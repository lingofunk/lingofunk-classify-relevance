#! /usr/bin/env python

import gc
import os
import numpy as np
import pandas as pd
from time import time

from bpemb import BPEmb
from keras.preprocessing import text, sequence
from scipy.spatial.distance import cosine

from keras.utils import Sequence

from utils import get_root


DIR_ROOT = get_root()
DIR_ASSETS = os.path.join(DIR_ROOT, "assets")
DATA_DIR = os.path.join(DIR_ROOT, "yelp-data")
PATH_TO_YELP_CSV = os.path.join(DATA_DIR, "restaurant_reviews.csv")

MAX_FEATURES = 100000
MAXLEN = 100

np.random.seed(42)


class Preprocess(object):
    def __init__(self, max_features, maxlen):
        self.max_features = max_features
        self.maxlen = maxlen
        self.tokenizer = text.Tokenizer(num_words=self.max_features)

    def fit_texts(self, list_sentences):
        self.tokenizer.fit_on_texts(list_sentences)

    def transform_texts(self, list_sentences):
        tokenized_sentences = self.tokenizer.texts_to_sequences(list_sentences)
        features = sequence.pad_sequences(tokenized_sentences, maxlen=self.maxlen)
        return features


def get_embeddings(word_index, max_features, embed_size):
    assert embed_size in [25, 50, 100, 200, 300]  # default sizes of embeddings in BPEmb
    bpemb_en = BPEmb(lang="en", dim=embed_size)
    embedding_matrix = np.zeros((max_features, embed_size))
    in_voc_words = 0

    for word, i in word_index.items():
        if i >= max_features:
            break
        in_voc_words += 1
        embedding_matrix[i] = np.sum(bpemb_en.embed(word), axis=0)

    print(f"{in_voc_words} words in vocabulary found out of {max_features} total.")

    return embedding_matrix


class YELPSequence(Sequence):
    def __init__(self, batch_size=128):
        super().__init__()
        self.restaurant_reviews = pd.read_csv(PATH_TO_YELP_CSV)
        self.restaurant_reviews = self.restaurant_reviews.groupby(["business_id"]).agg({"text": list})["text"].values
        self.n_comments_total = sum(len(restaurant) for restaurant in self.restaurant_reviews)
        self.n_restaurants = len(self.restaurant_reviews)
        self.lens_restaurants = np.array(list(map(len, self.restaurant_reviews)))
        self.batch_size = batch_size
        self.preprocessor = None

    def preprocess(self, preprocessor=None, preprocess=False):
        print("^^^^^^^^^^^^^")
        if (preprocessor is None) or preprocess:
            self.preprocessor = Preprocess(max_features=MAX_FEATURES, maxlen=MAXLEN)
            start = time()
            all_texts = sum(self.restaurant_reviews[:10000], [])
            print(type(all_texts[0]))
            print(len(all_texts), self.n_comments_total)
            self.preprocessor.fit_texts(all_texts)
            finish = time()
            print(f"{len(all_texts)} rests fitted", finish - start)
        else:
            self.preprocessor = preprocessor

    @staticmethod
    def compute_similarity(vec1, vec2):
        return cosine(vec1, vec2)

    def __len__(self):
        return int(np.ceil(self.n_restaurants / self.batch_size))

    def __getitem__(self, idx, process_target=True):
        idx_start = self.batch_size * idx
        idx_end = self.batch_size * (idx + 1)
        x_batch = []
        y_batch = []
        for i in range(idx_start, min(idx_end, self.n_restaurants)):
            n_comments = len(self.restaurant_reviews[i])
            probs = self.lens_restaurants / (self.n_comments_total - n_comments)
            probs[i] = 0

            # 1
            n_positive_examples = np.random.random_integers(2, max(3, n_comments // 10))
            positive_examples = np.random.random_integers(n_comments, size=(n_positive_examples, 2))
            for f, s in positive_examples:
                x_batch.append([self.restaurant_reviews[f][0], self.restaurant_reviews[s][0]])
            del positive_examples

            # 0
            n_negative_examples = 3 * np.random.random_integers(2, max(3, n_comments // 10))
            negative_restaurants = np.random.choice(self.n_restaurants, n_negative_examples, p=probs)
            negative_examples = []
            for restaurant in negative_restaurants:
                negative_examples.extend(np.random.choice(self.restaurant_reviews[restaurant]))
            del negative_restaurants
            restaurant_comment_embeddings = self.preprocessor.transform_texts(self.restaurant_reviews[i])
            negative_comment_embeddings = self.preprocessor.transform_texts(negative_examples)

            negative_pairs = []

            for j in range(n_negative_examples):
                comment_0_ind = np.random.randint(0, n_comments)
                negative_pairs.append((self.compute_similarity(restaurant_comment_embeddings[comment_0_ind],
                                                               negative_comment_embeddings[j]),
                                       comment_0_ind, j))
            negative_pairs.sort()
            kk = n_negative_examples // 3
            for k in range(kk):
                _, comment_0_ind, comment_1_ind = negative_pairs[k]
                x_batch.append([self.restaurant_reviews[i][comment_0_ind], negative_examples[comment_1_ind]])
            for k in range(2 * kk, n_negative_examples):
                _, comment_0_ind, comment_1_ind = negative_pairs[k]
                x_batch.append([self.restaurant_reviews[i][comment_0_ind], negative_examples[comment_1_ind]])
            y_batch.append([0] * (n_negative_examples - kk))

            del negative_pairs, negative_examples, restaurant_comment_embeddings, negative_comment_embeddings
            #gc.collect()
            #gc.collect()
            #gc.collect()
            #gc.collect()

            if process_target:
                y_batch = [1] * n_positive_examples + [0] * (n_negative_examples - kk)
                yield x_batch, y_batch
            else:
                yield x_batch
