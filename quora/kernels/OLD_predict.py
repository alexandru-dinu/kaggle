# coding: utf-8


import random
import re
import time
import os
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.tensor as tensor
import torch.utils.data
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import StratifiedKFold
from termcolor import colored
from tqdm.auto import tqdm

tqdm.pandas()

# -- globals

DATA_DIR = "../input"
TRAIN_CSV = f"{DATA_DIR}/train.csv"
TEST_CSV = f"{DATA_DIR}/test.csv"

EMB_GLOVE = f"{DATA_DIR}/embeddings/glove.840B.300d/glove.840B.300d.txt"
EMB_WORD2VEC = f"{DATA_DIR}/embeddings/GoogleNews-vectors-negative300/GoogleNews-vectors-negative300.bin"
EMB_PARAGRAM = f"{DATA_DIR}/embeddings/paragram_300_sl999/paragram_300_sl999.txt"
EMB_WIKI = f"{DATA_DIR}/embeddings/wiki-news-300d-1M/wiki-news-300d-1M.vec"

# -- globals


########################################################################################################################

print(colored("Cleaning data and building vocabulary", "yellow"))
_s = time.time()

train_df = pd.read_csv(TRAIN_CSV)
test_df = pd.read_csv(TEST_CSV)

SEP_PUNCTS = u'\u200b' + "/-'´‘…—−–"
SHOULD_KEEP_PUNCTS = "&"
TO_REMOVE_PUNCTS = '?!.,，"#$%\'()*+-/:;<=>@[\\]^_`{|}~“”’™•°'
# COMMON_WORDS       = ['the', 'a', 'to', 'of', 'and']

GLOVE_SYN_DICT = {
    'cryptocurrencies': 'crypto currencies',
    'quorans'         : 'quora members',
    'brexit'          : 'britain exit',
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


test_df["clean_question_text"] = test_df["question_text"].progress_apply(clean_all)
vocab = build_vocabulary(test_df["clean_question_text"])

_e = round(time.time() - _s, 5)
print(colored(f"Cleaned data: {_e}s\n", "green"))


########################################################################################################################


class Attention(nn.Module):
    def __init__(self, feature_dim, step_dim, bias=True, **kwargs):
        super(Attention, self).__init__(**kwargs)

        self.supports_masking = True

        self.bias = bias
        self.feature_dim = feature_dim
        self.step_dim = step_dim
        self.features_dim = 0

        weight = torch.zeros(feature_dim, 1)
        nn.init.xavier_uniform_(weight)
        self.weight = nn.Parameter(weight)

        if bias:
            self.b = nn.Parameter(torch.zeros(step_dim))

    def forward(self, x, mask=None):
        feature_dim = self.feature_dim
        step_dim = self.step_dim

        eij = torch.mm(
            x.contiguous().view(-1, feature_dim),
            self.weight
        ).view(-1, step_dim)

        if self.bias:
            eij = eij + self.b

        eij = torch.tanh(eij)
        a = torch.exp(eij)

        if mask is not None:
            a = a * mask

        a = a / torch.sum(a, 1, keepdim=True) + 1e-10

        weighted_input = x * torch.unsqueeze(a, -1)
        return torch.sum(weighted_input, 1)


class NeuralNet(nn.Module):
    def __init__(self, embedding_matrix):
        super(NeuralNet, self).__init__()

        hidden_size = 60
        maxlen = 60
        self.embedding_matrix = embedding_matrix

        # use pre-trained GloVe embeddings
        max_features, embed_size = self.embedding_matrix.shape
        self.embedding = nn.Embedding(max_features, embed_size)
        self.embedding.weight = nn.Parameter(torch.tensor(self.embedding_matrix, dtype=torch.float32))
        self.embedding.weight.requires_grad = False

        self.embedding_dropout = nn.Dropout2d(0.1)
        self.lstm = nn.GRU(embed_size, hidden_size, bidirectional=True, batch_first=True)
        self.gru = nn.GRU(hidden_size * 2, hidden_size, bidirectional=True, batch_first=True)

        self.lstm_attention = Attention(hidden_size * 2, maxlen)
        self.gru_attention = Attention(hidden_size * 2, maxlen)

        self.linear = nn.Linear(480, 16)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        self.out = nn.Linear(16, 1)

        self.sigmoid = lambda x: 1.0 / (1.0 + torch.exp(-x))

    def forward(self, x):
        h_embedding = self.embedding(x)
        h_embedding = torch.squeeze(self.embedding_dropout(torch.unsqueeze(h_embedding, 0)))
        # h_embedding = h_embedding.unsqueeze(0)

        h_lstm, _ = self.lstm(h_embedding)
        h_gru, _ = self.gru(h_lstm)

        h_lstm_atten = self.lstm_attention(h_lstm)
        h_gru_atten = self.gru_attention(h_gru)

        avg_pool = torch.mean(h_gru, 1)
        max_pool, _ = torch.max(h_gru, 1)

        conc = torch.cat((h_lstm_atten, h_gru_atten, avg_pool, max_pool), 1)
        conc = self.relu(self.linear(conc))
        conc = self.dropout(conc)
        out = self.out(conc)

        return out


########################################################################################################################


def make_data(df, len_voc, sentence_maxlen):
    t = Tokenizer(num_words=len_voc, filters='')
    t.fit_on_texts(df['clean_question_text'])

    X = pad_sequences(
        t.texts_to_sequences(df['clean_question_text']),
        maxlen=sentence_maxlen
    )

    return X


X = make_data(test_df, len_voc=len(vocab), sentence_maxlen=60)

########################################################################################################################

embedding_matrix = np.load("glove_embedding_matrix.npy")
print(colored(f"Loaded glove_embedding_matrix.npy\n", "green"))

from torchsummary import summary

checkpoint = "./checkpoints/first/first_model_f3_e5.pth"
model = NeuralNet(embedding_matrix=embedding_matrix)
model.cuda()
model.load_state_dict(torch.load(checkpoint))
model.eval()
print(colored(f"Loaded {checkpoint}\n", "green"))

summary(model, input_size=(1, 60)); exit(0)
#
qs = list(test_df['question_text'].values)
cqs = list(test_df['clean_question_text'].values)
# ts = list(test_df['target'].values)

while True:
    import random

    i = random.randint(0, len(X))
    x = torch.LongTensor(X[i, :]).unsqueeze(0).cuda()
    out = model(x)
    print(f"{i}: {qs[i]} -> {cqs[i]}")
    print(f"out: {out}\n")
    input()
