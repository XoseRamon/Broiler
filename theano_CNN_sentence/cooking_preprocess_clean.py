__author__ = "Urko Sanchez"

import cPickle
import json
import pandas as pd
import random
import sys
import timeit
from collections import defaultdict

import gensim
import numpy as np

reload(sys)
sys.setdefaultencoding('utf-8')

cuisine_classes = defaultdict(int)
ingredients_train = defaultdict(int)
ingredients_test = defaultdict(int)
ingredients_just_train = defaultdict(int)
ingredients_just_test = defaultdict(int)
ingredients_all = defaultdict(int)


# f = open('corpus.txt','w')


def pretty_print(d, indent=0):
    for key in sorted(d, key=d.get, reverse=True):
        print '\t' * indent + str(key) + " " + str(d[key])


with open('../test.json') as test_file:
    test_set = json.load(test_file)

with open('../train.json') as train_file:
    train_set = json.load(train_file)

# collects a dictionary class-frequency
# collects a dictionary ingredient-frequency for the train set
for item in train_set:
    cuisine_classes[item["cuisine"]] += 1
    for ingredient in item["ingredients"]:
        ingredients_train[ingredient.replace(" ", "_")] += 1
        ingredients_all[ingredient.replace(" ", "_")] += 1
        #     f.write(ingredient + " ")
        # f.write('\n')

# collects a dictionary ingredient-frequency for the test set
# collects a dictionary ingredient-frequency with ingredients JUST in test
for item in test_set:
    for ingredient in item["ingredients"]:
        if ingredient.replace(" ", "_") not in ingredients_train:
            ingredients_just_test[ingredient.replace(" ", "_")] += 1
        ingredients_test[ingredient.replace(" ", "_")] += 1
        ingredients_all[ingredient.replace(" ", "_")] += 1

# f.write(ingredient + " ")
#     f.write('\n')
# f.close()

# collects a dictionary ingredient-frequency with ingredients JUST in train
for item in train_set:
    for ingredient in item["ingredients"]:
        if ingredient.replace(" ", "_") not in ingredients_test:
            ingredients_just_train[ingredient.replace(" ", "_")] += 1


def get_W(word_vecs, k=300):
    """
    Get word matrix. W[i] is the vector for word indexed by i
    """
    vocab_size = len(word_vecs)
    word_idx_map = dict()
    W = np.zeros(shape=(vocab_size + 1, k))
    W[0] = np.zeros(k)
    i = 1
    for word in word_vecs:
        W[i] = word_vecs[word]
        word_idx_map[word] = i
        i += 1
    return W, word_idx_map


def load_model_vec(model, vocab):
    word_vecs = {}
    total_out = 0
    for line in vocab:
        if line not in model:
            total_out += 1
        else:
            word_vecs[line] = model[line]
    print "Not in model!!! %d" % total_out
    return word_vecs


def build_data_cv(train_set, cuisine_classes, frequent_ingredients, cv=10, shuffle=1):
    """
    Loads data and split into 10 folds.
    """
    data_cv = []
    # print "Number of shuffle %d" % shuffle

    for item in train_set:
        for i in range(shuffle + 1):
            ingredients = []
            random.shuffle(item["ingredients"])
            for ingredient in item["ingredients"]:
                if ingredient.replace(" ", "_") in frequent_ingredients:
                    ingredients.append(ingredient.replace(" ", "_"))
            datum = {"y": cuisine_classes.index(item["cuisine"]),
                     "recipe": " ".join(ingredients),
                     "num_ingredients": len(ingredients),
                     "split": np.random.randint(0, cv)}
            data_cv.append(datum)

    return data_cv


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


start = timeit.default_timer()


# use_classes : adds the target class of the recipe as a context word
def get_all_data(minfreq=1,window = 5, embed_dim = 300, shuffle = 1, use_classes = False):
    start_epoch = timeit.default_timer()
    print('-' * 40)
    print "Minimum freq %d" % minfreq
    print "Window size %d" % window
    print "Embedding dimension %d" % embed_dim
    print ('>' * 20)
    # prep for gensim format (change this, can be taken from before loops)
    # TODO: add minfreq here as well
    if (use_classes) :
        all_ingredients = [[ingredient.replace(" ", "_") for ingredient in item["ingredients"]]+[item["cuisine"]]
                   for item in train_set] + [ingredient.replace(" ", "_") for ingredient in item["ingredients"]
                   for item in test_set]
    else:
        all_ingredients = [[ingredient.replace(" ", "_") for ingredient in item["ingredients"] if
                            (ingredients_all[ingredient.replace(" ", "_")] > minfreq)]
                            for item in train_set + test_set]
    frequent_ingredients = [ingredient for ingredient in ingredients_all if
                            (ingredients_all[
                                 ingredient] > minfreq)]
    print "Number of items in train set %d" % sum(cuisine_classes.values())
    print "Number of classes of cuisine %d" % len(cuisine_classes)
    print "Number of ingredients in train %d" % len(ingredients_train)
    print "Number of ingredients JUST in train %d" % len(ingredients_just_train)
    print "Number of ingredients in test %d" % len(ingredients_test)
    print "Number of ingredients JUST in test %d" % len(ingredients_just_test)
    print "Number of ingredients in general %d" % len(frequent_ingredients)

    # pretty_print(ingredients_train)
    # print(dictionary.token2id)

    all_recipes = Recipe(all_ingredients)
    model = gensim.models.Word2Vec(all_recipes, size=embed_dim, min_count=1, hs=1, negative=0, window=window, iter=8)
    # model.save_word2vec_format("cooking_we_100d.txt")
    print "generating cv datasets..."
    cuisines = list(cuisine_classes.keys())
    # pprint(cuisine_classes.keys())
    data = build_data_cv(train_set, cuisines, frequent_ingredients, cv=10, shuffle=shuffle)
    max_l = np.max(pd.DataFrame(data)["num_ingredients"])
    print "number of recipes: " + str(len(data))
    print "vocab size: " + str(len(all_ingredients))
    print "max recipe length: " + str(max_l)
    print "loading word2vec vectors..."
    # This is probably unnecessary since the model already maps words and vectors
    w2v = load_model_vec(model, frequent_ingredients)
    print "word2vec loaded!"
    print "num words already in word2vec: " + str(len(w2v))
    W, word_idx_map = get_W(w2v, embed_dim)
    rand_vecs = {}
    add_unknown_words(rand_vecs, ingredients_all, k=embed_dim)
    W2, _ = get_W(rand_vecs, k=embed_dim)
    stop = timeit.default_timer()
    print "Time epoch %d seconds" % (stop - start_epoch)
    print "Time accumulated %d seconds" % (stop - start)
    print('-' * 40)
    return data, W, W2, word_idx_map, ingredients_all, cuisines


# print "Number of ingredients in train and test %d" % (len(ingredients_train) + len(ingredients_test))
# print str((float(len(ingredients_test))*100 / (len(ingredients_train) + len(ingredients_test)))) + "%"


# Histogram of classes
# X = np.arange(len(cuisine_classes))
# pl.bar(X, sorted(cuisine_classes.values(), reverse=True), width=0.3)
# pl.xticks(X, sorted(cuisine_classes, key=cuisine_classes.get, reverse=True))
# ymax = max(cuisine_classes.values()) + 100
# pl.ylim(0, ymax)
# pl.show()

# main_ingredients = {k:v for (k,v) in ingredients_train.items() if v > 4000}
# print len(main_ingredients)
# # Histogram of ingredients
# X = np.arange(len(main_ingredients))
# pl.bar(X, sorted(main_ingredients.values(), reverse=True), width=0.3)
# pl.xticks(X, sorted(main_ingredients, key=main_ingredients.get, reverse=True))
# ymax = max(main_ingredients.values()) + 100
# pl.ylim(0, ymax)
# pl.show()
