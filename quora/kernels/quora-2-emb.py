# coding: utf-8


import re
import time
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.tensor as tensor
import torch.utils.data
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from torch.autograd import Variable
from tqdm.auto import tqdm

tqdm.pandas()

assert torch.cuda.is_available()

# GLOBALS ##############################################################################################################

DATA_DIR = "../input"
TRAIN_CSV = f"{DATA_DIR}/train.csv"
TEST_CSV = f"{DATA_DIR}/test.csv"

EMB_GLOVE = f"{DATA_DIR}/embeddings/glove.840B.300d/glove.840B.300d.txt"
EMB_WORD2VEC = f"{DATA_DIR}/embeddings/GoogleNews-vectors-negative300/GoogleNews-vectors-negative300.bin"
EMB_PARAGRAM = f"{DATA_DIR}/embeddings/paragram_300_sl999/paragram_300_sl999.txt"
EMB_WIKI = f"{DATA_DIR}/embeddings/wiki-news-300d-1M/wiki-news-300d-1M.vec"

# CLEANING #############################################################################################################

SEP_PUNCTS = u'\u200b' + "/-'´‘…—−–"
SHOULD_KEEP_PUNCTS = "&"
TO_REMOVE_PUNCTS = '?!.,，"#$%\'()*+-/:;<=>@[\\]^_`{|}~“”’™•°'
GLOVE_SYN_DICT = {
    'cryptocurrencies': 'crypto currencies',
    'ethereum'        : 'crypto currency',
    'fortnite'        : 'video game',
    'quorans'         : 'quora members',
    'brexit'          : 'britain exit',
    'redmi'           : 'xiaomi',
    '√'               : 'square root',
    '÷'               : 'division',
    '∞'               : 'infinity',
    '€'               : 'euro',
    '£'               : 'pound sterling',
    '$'               : 'dollar',
    '₹'               : 'rupee',
    '×'               : 'product',
    'ã'               : 'a',
    'è'               : 'e',
    'é'               : 'e',
    'ö'               : 'o',
    '²'               : 'squared',
    '∈'               : 'in',
    '∩'               : 'intersection',
    u'\u0398'         : 'Theta',
    u'\u03A0'         : 'Pi',
    u'\u03A9'         : 'Omega',
    u'\u0392'         : 'Beta',
    u'\u03B8'         : 'theta',
    u'\u03C0'         : 'pi',
    u'\u03C9'         : 'omega',
    u'\u03B2'         : 'beta',
}


def tokenize(s: str):
    return list(map(lambda w: w.strip(), s.split()))


def clean_text(x):
    x = x.lower()

    for p in SEP_PUNCTS:
        x = x.replace(p, " ")
    for p in SHOULD_KEEP_PUNCTS:
        x = x.replace(p, f" {p} ")
    for p in TO_REMOVE_PUNCTS:
        x = x.replace(p, "")

    return x


def clean_numbers(x):
    x = re.sub('[0-9]{5,}', '#####', x)
    x = re.sub('[0-9]{4}', '####', x)
    x = re.sub('[0-9]{3}', '###', x)
    x = re.sub('[0-9]{2}', '##', x)

    return x


def clean_syn(x):
    regex = re.compile('(%s)' % '|'.join(GLOVE_SYN_DICT.keys()))
    return regex.sub(lambda m: GLOVE_SYN_DICT.get(m.group(0), ''), x)


def clean_all(x):
    x = clean_text(x)
    x = clean_syn(x)
    return x


def build_vocabulary(df: pd.DataFrame) -> Counter:
    sentences = df.progress_apply(tokenize).values
    vocab = Counter()

    for sentence in tqdm(sentences):
        for word in sentence:
            vocab[word] += 1
    return vocab


# LOAD DATA ############################################################################################################

def make_data(df, len_voc, sentence_maxlen, is_test=False):
    t = Tokenizer(num_words=len_voc, filters='')
    t.fit_on_texts(df['clean_question_text'])

    X = pad_sequences(sequences=t.texts_to_sequences(df['clean_question_text']), maxlen=sentence_maxlen)

    if not is_test:
        Y = df['target'].values
        return X, Y, t.word_index
    else:
        return X


def load_glove():
    def get_coefs(word, *arr):
        return word, np.asarray(arr, dtype='float32')

    return dict(get_coefs(*o.split(" ")) for o in open(EMB_GLOVE, encoding='latin'))


def make_embedding_matrix(embeddings_dict, w_idx, len_voc):
    """
    Random values of oov words
    """
    all_embs = np.stack(list(embeddings_dict.values()))
    emb_mean, emb_std = all_embs.mean(), all_embs.std()
    embed_size = all_embs.shape[1]

    _embedding_matrix = np.random.normal(emb_mean, emb_std, (len_voc, embed_size))

    for word, wi in tqdm(w_idx.items()):
        if wi >= len_voc:
            continue
        embedding_vector = embeddings_dict.get(word, None)
        if embedding_vector is not None:
            _embedding_matrix[wi] = embedding_vector

    return _embedding_matrix


# MODEL ################################################################################################################

class Net(nn.Module):
    def __init__(self, emb_matrix, sen_maxlen, num_layers=1):
        super(Net, self).__init__()

        # GloVe emb matrix
        num_words, emb_size = emb_matrix.shape

        self.hidden_size = emb_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(num_words, emb_size)
        self.embedding.weight = nn.Parameter(torch.tensor(emb_matrix, dtype=torch.float32))
        self.embedding.weight.requires_grad = False

        self.gru = nn.GRU(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            num_layers=1,
            batch_first=True
        )

        self.fc = nn.Linear(self.hidden_size, 1)

        self.droput = nn.Dropout(0.1)
        self.relu = nn.ReLU()

    def forward(self, x):
        # x: B x sen_maxlen

        emb = self.embedding(x)
        # B x sen_maxlen x emb_size

        hidden = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size).cuda())
        out, hidden = self.gru(emb, hidden)
        # hidden: num_layers x B x emb_size

        out = self.fc(hidden)
        # out: num_layers x B x 1

        return out


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


# MAIN #################################################################################################################

# HP
SENTENCE_MAXLEN = 35
BATCH_SIZE = 256
NUM_EPOCHS = 5
LOCAL = True
# --

print("Loading csvs")
TRAIN_DF = pd.read_csv(TRAIN_CSV)
TEST_DF = pd.read_csv(TEST_CSV)

print("Building cleaned vocabs")
TRAIN_DF["clean_question_text"] = TRAIN_DF["question_text"].progress_apply(clean_all)
TEST_DF["clean_question_text"] = TEST_DF["question_text"].progress_apply(clean_all)

TRAIN_VOCAB = build_vocabulary(TRAIN_DF["clean_question_text"])
TEST_VOCAB = build_vocabulary(TEST_DF["clean_question_text"])

print("Make train/test data")
# X = train_size x sentence_maxlen
X_TRAIN, Y_TRAIN, WORD_INDEX = make_data(TRAIN_DF, len_voc=len(TRAIN_VOCAB), sentence_maxlen=SENTENCE_MAXLEN)
X_TEST = make_data(TEST_DF, len_voc=len(TEST_VOCAB), sentence_maxlen=SENTENCE_MAXLEN, is_test=True)

print("Load emb matrix")
if LOCAL:
    EMBEDDING_MATRIX = np.load("glove_embedding_matrix.npy")
else:
    EMBEDDING_MATRIX = make_embedding_matrix(embeddings_dict=load_glove(), w_idx=WORD_INDEX, len_voc=len(TRAIN_VOCAB))

print("\nStarting train loop\n")

SEED = None
train_splits = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED).split(X_TRAIN, Y_TRAIN))

train_preds = np.zeros((len(X_TRAIN)))
test_preds = np.zeros((len(X_TEST)))

test_dataloader = torch.utils.data.DataLoader(
    dataset=torch.utils.data.TensorDataset(torch.tensor(X_TEST, dtype=torch.long).cuda()),
    batch_size=BATCH_SIZE, shuffle=False
)

for fold_idx, (train_idx, val_idx) in enumerate(train_splits, start=1):

    x_train_fold = torch.tensor(X_TRAIN[train_idx], dtype=torch.long).cuda()
    y_train_fold = torch.tensor(Y_TRAIN[train_idx, np.newaxis], dtype=torch.float32).cuda()
    x_val_fold = torch.tensor(X_TRAIN[val_idx], dtype=torch.long).cuda()
    y_val_fold = torch.tensor(Y_TRAIN[val_idx, np.newaxis], dtype=torch.float32).cuda()

    # TODO: change sen maxlen maybe?
    model = Net(emb_matrix=EMBEDDING_MATRIX, sen_maxlen=SENTENCE_MAXLEN, num_layers=1)
    model.cuda()

    loss_fn = torch.nn.BCEWithLogitsLoss(reduction="sum")
    # loss_fn = torch.nn.CrossEntropyLoss(reduction="sum")

    # TODO: schedule?
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

    train_dataloader = torch.utils.data.DataLoader(
        dataset=torch.utils.data.TensorDataset(x_train_fold, y_train_fold), batch_size=BATCH_SIZE, shuffle=True
    )
    val_dataloader = torch.utils.data.DataLoader(
        dataset=torch.utils.data.TensorDataset(x_val_fold, y_val_fold), batch_size=BATCH_SIZE, shuffle=False
    )

    test_preds_fold = np.zeros(len(X_TEST))
    val_preds_fold = np.zeros((x_val_fold.size(0)))

    num_batches = len(train_dataloader)
    print(f"Fold {fold_idx}; num_batches = {num_batches}")

    # for current fold, train NUM_EPOCHS
    for epoch_idx in range(1, NUM_EPOCHS + 1):
        start_time = time.time()

        model.train()
        avg_loss = 0.0
        my_loss = 0.0

        print(f"Epoch {epoch_idx}/{NUM_EPOCHS}")
        for batch_idx, (x_batch, y_batch) in tqdm(enumerate(train_dataloader, start=1), total=len(train_dataloader)):
            y_pred = model(x_batch).squeeze(0)

            loss = loss_fn(y_pred, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            avg_loss += loss.item() / len(train_dataloader)
            my_loss += loss.item()

            if batch_idx % 512 == 0:
                print("loss:", my_loss / 512.0)
                my_loss = 0.0
        # -- end batch

        print("Cross-validation")
        model.eval()
        avg_val_loss = 0.0
        for i, (x_batch, y_batch) in tqdm(enumerate(val_dataloader), total=len(val_dataloader)):
            y_pred = model(x_batch).detach().squeeze(0)
            avg_val_loss += loss_fn(y_pred, y_batch).item() / len(val_dataloader)
            val_preds_fold[i * BATCH_SIZE:(i + 1) * BATCH_SIZE] = sigmoid(y_pred.cpu().numpy())[:, 0]
        # --

        print('\n\nsummary: Fold {}/{} \t Epoch {}/{} \t loss={:.4f} \t val_loss={:.4f} \t time={:.2f}s\n'.format(
            fold_idx, len(train_splits), epoch_idx, NUM_EPOCHS, avg_loss, avg_val_loss, time.time() - start_time)
        )

        if LOCAL:
            _w_name = f"./checkpoints/slen35_model_f{fold_idx}_e{epoch_idx}.pth"
            torch.save(model.state_dict(), _w_name)
            print(f"Saved weights to [{_w_name}]\n")
    # -- end epoch

    # for current fold
    print(f"Fold {fold_idx} done; test on test data")
    for i, (x_batch,) in tqdm(enumerate(test_dataloader), total=len(test_dataloader)):
        y_pred = model(x_batch).detach().squeeze(0)
        test_preds_fold[i * BATCH_SIZE:(i + 1) * BATCH_SIZE] = sigmoid(y_pred.cpu().numpy())[:, 0]

    # fill predictions for training data from current validation set
    train_preds[val_idx] = val_preds_fold
    # average predictions for each split
    test_preds += test_preds_fold / len(train_splits)


# SUBMIT ###############################################################################################################

def threshold_search(y_true, y_predicted):
    best_threshold, best_score = 0, 0

    for thr in tqdm(np.linspace(-1.5, 1.5, 75)):
        score = f1_score(y_true=y_true, y_pred=(y_predicted > thr).astype(int))
        if score > best_score:
            best_threshold = thr
            best_score = score

    return {'threshold': best_threshold, 'f1': best_score}


search_result = threshold_search(Y_TRAIN, train_preds)
print(search_result)

submission = pd.DataFrame.from_dict({'qid': TEST_DF['qid']})
submission['prediction'] = (test_preds > search_result['threshold']).astype(int)
submission.to_csv("submission.csv", index=False)
