# coding: utf-8


import datetime
import re
import time
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from tqdm.auto import tqdm

tqdm.pandas()

assert torch.cuda.is_available()

# GLOBALS ##############################################################################################################

DATA_DIR = "../input"
TRAIN_CSV = f"{DATA_DIR}/train.csv"
TEST_CSV = f"{DATA_DIR}/test.csv"

EMB_GLOVE_FILE = f"{DATA_DIR}/embeddings/glove.840B.300d/glove.840B.300d.txt"
EMB_PARAGRAM_FILE = f"{DATA_DIR}/embeddings/paragram_300_sl999/paragram_300_sl999.txt"

LOG_NAME = "./output.log"
LOG_FP = open(LOG_NAME, "wt")


def LOG(*s):
    _s = str(datetime.datetime.now()).split(".")[0] + str(s) + "\n"
    print(_s)
    LOG_FP.write(_s)


# CLEANING #############################################################################################################

PUNCTUATION = {
    "sep": "\u200b" + "/-'´′‘…—−–",
    "keep": "&",
    "remove": "?!.,，\"#$%'()*+-/:;<=>@[\\]^_`{|}~“”’™•°",
}

GLOVE_SYN_DICT = {
    "cryptocurrencies": "crypto currencies",
    "ethereum": "crypto currency",
    "fortnite": "video game",
    "quorans": "quora members",
    "brexit": "britain exit",
    "redmi": "xiaomi",
    "√": "square root",
    "÷": "division",
    "∞": "infinity",
    "€": "euro",
    "£": "pound sterling",
    "$": "dollar",
    "₹": "rupee",
    "×": "product",
    "ã": "a",
    "è": "e",
    "é": "e",
    "ö": "o",
    "²": "squared",
    "∈": "in",
    "∩": "intersection",
    "\u0398": "Theta",
    "\u03A0": "Pi",
    "\u03A9": "Omega",
    "\u0392": "Beta",
    "\u03B8": "theta",
    "\u03C0": "pi",
    "\u03C9": "omega",
    "\u03B2": "beta",
}


def tokenize(s: str):
    return list(map(lambda w: w.strip(), s.split()))


def clean_text(x):
    x = x.lower()

    for p in PUNCTUATION["sep"]:
        x = x.replace(p, " ")
    for p in PUNCTUATION["keep"]:
        x = x.replace(p, f" {p} ")
    for p in PUNCTUATION["remove"]:
        x = x.replace(p, "")

    return x


def clean_numbers(x):
    x = re.sub("[0-9]{5,}", "#####", x)
    x = re.sub("[0-9]{4}", "####", x)
    x = re.sub("[0-9]{3}", "###", x)
    x = re.sub("[0-9]{2}", "##", x)

    return x


def clean_syn(x):
    regex = re.compile("(%s)" % "|".join(GLOVE_SYN_DICT.keys()))
    return regex.sub(lambda m: GLOVE_SYN_DICT.get(m.group(0), ""), x)


def clean_all(x):
    x = clean_text(x)
    x = clean_syn(x)
    return x


# LOAD DATA ############################################################################################################


def build_glove_embedding_matrix(w_idx, len_voc):
    def get_coefs(w, *arr):
        return w, np.asarray(arr, dtype="float32")

    emb_dict = dict(
        get_coefs(*o.split(" ")) for o in open(EMB_GLOVE_FILE, encoding="latin")
    )

    all_embs = np.stack(list(emb_dict.values()))
    embed_size = all_embs.shape[1]

    n_words = min(len_voc, len(w_idx))
    # emb_mean, emb_std = all_embs.mean(), all_embs.std()
    # emb_matrix = np.random.normal(emb_mean, emb_std, (n_words, embed_size))
    emb_matrix = np.zeros((n_words, embed_size))

    for word, wi in w_idx.items():
        if wi >= len_voc:
            continue

        emb_vector = emb_dict.get(word, None)
        if emb_vector is not None:
            emb_matrix[wi] = emb_vector

    return emb_matrix


def build_paragram_embedding_matrix(w_idx, len_voc):
    def get_coefs(w, *arr):
        return w, np.asarray(arr, dtype="float32")

    emb_dict = dict(
        get_coefs(*o.split(" "))
        for o in open(EMB_PARAGRAM_FILE, encoding="utf8", errors="ignore")
        if len(o) > 100
    )

    all_embs = np.stack(emb_dict.values())
    embed_size = all_embs.shape[1]

    n_words = min(len_voc, len(w_idx))
    # emb_mean, emb_std = all_embs.mean(), all_embs.std()
    # emb_matrix = np.random.normal(emb_mean, emb_std, (n_words, embed_size))
    emb_matrix = np.zeros((n_words, embed_size))

    for word, wi in w_idx.items():
        if wi >= len_voc:
            continue

        emb_vector = emb_dict.get(word, None)
        if emb_vector is not None:
            emb_matrix[wi] = emb_vector

    return emb_matrix


def build_vocabulary(df: pd.DataFrame) -> Counter:
    sentences = df.progress_apply(tokenize).values
    vocab = Counter()

    for sentence in tqdm(sentences):
        for word in sentence:
            vocab[word] += 1

    return vocab


def load_data(sentence_maxlen, shuffle_train=False):
    LOG("Read csvs...")
    train_df = pd.read_csv(TRAIN_CSV)
    test_df = pd.read_csv(TEST_CSV)

    LOG("Clean DataFrames...")
    train_df["question_text"] = train_df["question_text"].progress_apply(clean_all)
    test_df["question_text"] = test_df["question_text"].progress_apply(clean_all)

    train_vocab = build_vocabulary(train_df["question_text"])
    num_words = len(train_vocab)

    X_train = train_df["question_text"].fillna("_##_").values
    X_test = test_df["question_text"].fillna("_##_").values

    LOG("Tokenize...")
    tok = Tokenizer(num_words=num_words)
    tok.fit_on_texts(list(X_train))

    LOG("Pad...")
    X_train = pad_sequences(tok.texts_to_sequences(X_train), maxlen=sentence_maxlen)
    X_test = pad_sequences(tok.texts_to_sequences(X_test), maxlen=sentence_maxlen)
    Y_train = train_df["target"].values

    word_index = tok.word_index

    if shuffle_train:
        idx = np.random.permutation(len(X_train))
        X_train = X_train[idx]
        Y_train = Y_train[idx]

    LOG("Embedding matrices")
    e1 = build_glove_embedding_matrix(w_idx=word_index, len_voc=len(train_vocab))
    LOG("->loaded glove")
    e2 = build_paragram_embedding_matrix(w_idx=word_index, len_voc=len(train_vocab))
    LOG("->loaded paragram")
    emb_matrix = np.mean([e1, e2], axis=0)

    return X_train, Y_train, X_test, train_vocab, emb_matrix, word_index


# MODEL ################################################################################################################


class Net(nn.Module):
    def __init__(self, emb_matrix, hidden_size):
        super(Net, self).__init__()

        num_words, emb_size = emb_matrix.shape

        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(num_words, emb_size)
        self.embedding.weight = nn.Parameter(
            torch.tensor(emb_matrix, dtype=torch.float32)
        )
        self.embedding.weight.requires_grad = False

        self.lstm = nn.LSTM(
            input_size=emb_size,
            hidden_size=self.hidden_size,
            num_layers=1,
            bidirectional=True,
            batch_first=True,
        )

        self.gru = nn.GRU(
            input_size=2 * self.hidden_size,
            hidden_size=self.hidden_size,
            num_layers=1,
            bidirectional=True,
            batch_first=True,
        )

        self.fc = nn.Linear(2 * self.hidden_size, 1)

        self.dropout = nn.Dropout(0.1)
        self.relu = nn.ReLU()

    def forward(self, x):
        # x: B x sen_maxlen

        emb = self.embedding(x)
        # B x sen_maxlen x emb_size

        out_lstm, _ = self.lstm(emb)
        # B x sen_maxlen x (2*sen_maxlen)

        _, h_gru = self.gru(self.dropout(out_lstm))
        # 2 x B x sen_maxlen

        h_gru = h_gru.permute((1, 0, 2)).reshape(x.size(0), -1)
        # B x (2*sen_maxlen)

        out = self.fc(h_gru).unsqueeze(0)
        # 1 x B x 1

        return out


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


# MAIN #################################################################################################################


LOCAL = False  # whether it's running locally or on kaggle

HP = {
    "sentence_maxlen": 70,
    "batch_size": 256,
    "num_epochs": 8,
}

# LOAD DATA
X_TRAIN, Y_TRAIN, X_TEST, TRAIN_VOCAB, EMBEDDING_MATRIX, WORD_INDEX = load_data(
    sentence_maxlen=HP["sentence_maxlen"], shuffle_train=True
)

LOG("\nStarting train loop\n")

SEED = None
train_splits = list(
    StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED).split(X_TRAIN, Y_TRAIN)
)

train_preds = np.zeros((len(X_TRAIN)))
test_preds = np.zeros((len(X_TEST)))

test_dataloader = torch.utils.data.DataLoader(
    dataset=torch.utils.data.TensorDataset(
        torch.tensor(X_TEST, dtype=torch.long).cuda()
    ),
    batch_size=HP["batch_size"],
    shuffle=False,
)

for fold_idx, (train_idx, val_idx) in enumerate(train_splits, start=1):

    x_train_fold = torch.tensor(X_TRAIN[train_idx], dtype=torch.long).cuda()
    y_train_fold = torch.tensor(
        Y_TRAIN[train_idx, np.newaxis], dtype=torch.float32
    ).cuda()
    x_val_fold = torch.tensor(X_TRAIN[val_idx], dtype=torch.long).cuda()
    y_val_fold = torch.tensor(Y_TRAIN[val_idx, np.newaxis], dtype=torch.float32).cuda()

    model = Net(emb_matrix=EMBEDDING_MATRIX, hidden_size=HP["sentence_maxlen"])
    model.cuda()

    loss_fn = torch.nn.BCEWithLogitsLoss(reduction="sum")
    # loss_fn = torch.nn.CrossEntropyLoss(reduction="sum")

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)

    train_dataloader = torch.utils.data.DataLoader(
        dataset=torch.utils.data.TensorDataset(x_train_fold, y_train_fold),
        batch_size=HP["batch_size"],
        shuffle=True,
    )
    val_dataloader = torch.utils.data.DataLoader(
        dataset=torch.utils.data.TensorDataset(x_val_fold, y_val_fold),
        batch_size=HP["batch_size"],
        shuffle=False,
    )

    test_preds_fold = np.zeros(len(X_TEST))
    val_preds_fold = np.zeros((x_val_fold.size(0)))

    num_batches = len(train_dataloader)
    LOG(f"Fold {fold_idx}; num_batches = {num_batches}")

    # for current fold, train NUM_EPOCHS
    for epoch_idx in range(1, HP["num_epochs"] + 1):
        start_time = time.time()

        model.train()
        avg_loss = 0.0
        my_loss = 0.0

        lr_scheduler.step()

        LOG(f"Epoch {epoch_idx}/{HP['num_epochs']}; lr = {lr_scheduler.get_lr()}")
        for batch_idx, (x_batch, y_batch) in tqdm(
            enumerate(train_dataloader, start=1), total=len(train_dataloader)
        ):
            y_pred = model(x_batch).squeeze(0)

            loss = loss_fn(y_pred, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            avg_loss += loss.item() / len(train_dataloader)
            my_loss += loss.item()

            if batch_idx % 512 == 0:
                LOG("->loss:", my_loss / 512.0)
                my_loss = 0.0
        # -- end batch

        LOG("Cross-validation")
        model.eval()
        avg_val_loss = 0.0

        for i, (x_batch, y_batch) in tqdm(
            enumerate(val_dataloader), total=len(val_dataloader)
        ):
            y_pred = model(x_batch).detach().squeeze(0)
            avg_val_loss += loss_fn(y_pred, y_batch).item() / len(val_dataloader)
            val_preds_fold[i * HP["batch_size"] : (i + 1) * HP["batch_size"]] = sigmoid(
                y_pred.cpu().numpy()
            )[:, 0]
        # --

        LOG(
            "\n\nsummary: Fold {}/{} \t Epoch {}/{} \t loss={:.4f} \t val_loss={:.4f} \t time={:.2f}s\n".format(
                fold_idx,
                len(train_splits),
                epoch_idx,
                HP["num_epochs"],
                avg_loss,
                avg_val_loss,
                time.time() - start_time,
            )
        )
    # -- end epoch

    # for current fold, predict on test data
    LOG(f"Fold {fold_idx} done; test on test data")
    for i, (x_batch,) in tqdm(enumerate(test_dataloader), total=len(test_dataloader)):
        y_pred = model(x_batch).detach().squeeze(0)
        test_preds_fold[i * HP["batch_size"] : (i + 1) * HP["batch_size"]] = sigmoid(
            y_pred.cpu().numpy()
        )[:, 0]
    # --

    train_preds[
        val_idx
    ] = val_preds_fold  # fill predictions for training data from current validation set
    test_preds += test_preds_fold / len(
        train_splits
    )  # average test predictions for each fold

LOG("Training done")


# SUBMIT ###############################################################################################################


def threshold_search(y_true, y_predicted):
    best_threshold, best_score = 0, 0

    for thr in tqdm(np.linspace(-1.5, 1.5, 100)):
        score = f1_score(y_true=y_true, y_pred=(y_predicted > thr).astype(int))
        if score > best_score:
            best_threshold = thr
            best_score = score

    return {"threshold": best_threshold, "f1": best_score}


LOG("Finding threshold")
search_result = threshold_search(Y_TRAIN, train_preds)
LOG(search_result)

LOG("Generating submission.csv")
submission = pd.read_csv("../input/sample_submission.csv")
submission.prediction = (test_preds > search_result["threshold"]).astype(int)
submission.to_csv("submission.csv", index=False)

LOG_FP.close()
