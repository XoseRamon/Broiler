from __future__ import absolute_import
import json
import re
from collections import defaultdict
import numpy as np
import random
from gensim import models


def load_data(test_split=0.0, shuffle=1):
    """
    Loads data and split into 10 folds.
    """
    with open('../test.json') as test_file:
        test_set = json.load(test_file)

    with open('../train.json') as train_file:
        train_set = json.load(train_file)

    ingredients_train = defaultdict(int)
    cuisine_classes = defaultdict(int)
    X = []
    X_prediction = []
    X_id = []
    labels = []
    for item in train_set:
        cuisine_classes[item["cuisine"]] += 1
        for ingredient in item["ingredients"]:
            ingredients_train[ingredient.replace(" ", "_")] += 1

    vocab = list(ingredients_train.keys())
    cuisines = list(cuisine_classes.keys())
    for item in train_set:
        for i in range(shuffle):
            random.shuffle(item["ingredients"])
            X.append([vocab.index(ingredient.replace(" ", "_")) for ingredient in item["ingredients"]])
            labels.append(cuisines.index(item["cuisine"]))

    for item in test_set:
        random.shuffle(item["ingredients"])
        X_prediction.append([vocab.index(ingredient.replace(" ", "_")) for ingredient in item["ingredients"] if
                             ingredient.replace(" ", "_") in vocab])
        X_id.append(item["id"])

    X_train = X[:int(len(X) * (1 - test_split))]
    y_train = labels[:int(len(X) * (1 - test_split))]

    X_test = X[int(len(X) * (1 - test_split)):]
    y_test = labels[int(len(X) * (1 - test_split)):]
    return (X_train, y_train), (X_test, y_test), (X_prediction, X_id), vocab, cuisines

def load_full_data(shuffle=1):
    with open('../test.json') as test_file:
        test_set = json.load(test_file)

    with open('../train.json') as train_file:
        train_set = json.load(train_file)

    ingredients_train = defaultdict(int)
    cuisine_classes = defaultdict(int)
    X = []
    X_prediction = []
    X_id = []
    labels = []
    for item in train_set:
        cuisine_classes[item["cuisine"]] += 1
        for ingredient in item["ingredients"]:
            ingredients_train[ingredient.replace(" ", "_")] += 1

    vocab = list(ingredients_train.keys())
    cuisines = list(cuisine_classes.keys())
    for item in train_set:
        for i in range(shuffle):
            random.shuffle(item["ingredients"])
            X.append([vocab.index(ingredient.replace(" ", "_")) for ingredient in item["ingredients"]])
            labels.append(cuisines.index(item["cuisine"]))

    for item in test_set:
        random.shuffle(item["ingredients"])
        X_prediction.append([vocab.index(ingredient.replace(" ", "_")) for ingredient in item["ingredients"] if
                             ingredient.replace(" ", "_") in vocab])
        X_id.append(item["id"])

    return (X, labels), (X_prediction, X_id), vocab, cuisines


# def get_W(word_vecs, k=300):
#     """
#     Get word matrix. W[i] is the vector for word indexed by i
#     """
#     vocab_size = len(word_vecs)
#     word_idx_map = dict()
#     W = np.zeros(shape=(vocab_size + 1, k))
#     W[0] = np.zeros(k)
#     i = 1
#     for word in word_vecs:
#         W[i] = word_vecs[word]
#         word_idx_map[word] = i
#         i += 1
#     return W, word_idx_map


def load_bin_vec(fname, vocab):
    """
    Loads 300x1 word vecs from Google (Mikolov) word2vec
    """
    word_vecs = {}
    with open(fname, "rb") as f:
        header = f.readline()
        vocab_size, layer1_size = map(int, header.split())
        binary_len = np.dtype('float32').itemsize * layer1_size
        for line in xrange(vocab_size):
            word = []
            while True:
                ch = f.read(1)
                if ch == ' ':
                    word = ''.join(word)
                    break
                if ch != '\n':
                    word.append(ch)
            if word in vocab:
                hey = np.fromstring(f.read(binary_len), dtype='float32')
                print type(hey)
                word_vecs[word] = hey
            else:
                f.read(binary_len)
    return word_vecs


def add_unknown_words(word_vecs, vocab, min_df=1, k=300):
    """
    For words that occur in at least min_df documents, create a separate word vector.
    0.25 is chosen so the unknown vectors have (approximately) same variance as pre-trained ones
    """
    for word in vocab:
        if word not in word_vecs and vocab[word] >= min_df:
            word_vecs[word] = np.random.uniform(-0.25, 0.25, k)

class Recipe(object):
    def __init__(self, list_recipes):
        self.list_recipes = list_recipes

    def __iter__(self):
        for recipe in self.list_recipes:
            yield recipe

def generateWord2Vec(vocab, size=300, window=70, use_classes=True):
    with open('../test.json') as test_file:
        test_set = json.load(test_file)

    with open('../train.json') as train_file:
        train_set = json.load(train_file)

    all_ingredients = defaultdict(int)

    if (use_classes) :
        all_ingredients = [[ingredient.replace(" ", "_") for ingredient in item["ingredients"]]+[item["cuisine"]]
                   for item in train_set] + [[ingredient.replace(" ", "_") for ingredient in item["ingredients"]]
                   for item in test_set]
    else:
        all_ingredients = [[ingredient.replace(" ", "_") for ingredient in item["ingredients"]]
                            for item in train_set + test_set]

    all_recipes = Recipe(all_ingredients)
    model = models.Word2Vec(all_recipes, size=size, min_count=1, hs=1, negative=0, window=window, iter=10)

    #Transform model into expected format and add random missing words
    word_vecs = {}
    for line in vocab:
        if line in model:
            word_vecs[line] = model[line]
        else:
            word_vecs[line] = np.random.uniform(-0.25, 0.25, size)
    return word_vecs

def clean_str(string, TREC=False):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Every dataset is lower cased except for TREC
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip() if TREC else string.strip().lower()


def clean_str_sst(string):
    """
    Tokenization/string cleaning for the SST dataset
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()