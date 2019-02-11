import os

import numpy as np
import pandas as pd
from bpemb import BPEmb
from keras.preprocessing import sequence, text
from keras.utils import Sequence
from scipy.spatial.distance import cosine
from sklearn.utils import shuffle as skshuffle

from lingofunk_classify_relevance.config import fetch_data
from lingofunk_classify_relevance.data.traintest_generator import split_data

PATH_TO_YELP_CSV_TRAIN = fetch_data("train")
PATH_TO_YELP_CSV_TEST = fetch_data("test")

MAX_FEATURES = 100_000
MAXLEN = 100

np.random.seed(42)


class Preprocess:
    def __init__(self, max_features, maxlen):
        self.max_features = max_features
        self.maxlen = maxlen
        self.tokenizer = text.Tokenizer(num_words=self.max_features)

    def fit_texts(self, list_sentences):
        self.tokenizer.fit_on_texts(list_sentences)

    def transform_texts(self, list_sentences):
        # print("TYPE: ", type(list_sentences), type(list_sentences[0]))
        tokenized_sentences = self.tokenizer.texts_to_sequences(list_sentences)
        features = sequence.pad_sequences(tokenized_sentences, maxlen=self.maxlen)
        return features


class YELPSequence(Sequence):
    def __init__(self, batch_size=128, test=False, preprocessor=None):
        super().__init__()

        trainDoesExist = os.path.isfile(PATH_TO_YELP_CSV_TRAIN)
        testDoesExist = os.path.isfile(PATH_TO_YELP_CSV_TEST)

        if not (trainDoesExist and testDoesExist):
            split_data()

        if test:
            self.restaurant_reviews = pd.read_csv(PATH_TO_YELP_CSV_TEST)
        else:
            self.restaurant_reviews = pd.read_csv(PATH_TO_YELP_CSV_TRAIN)

        self.restaurant_reviews = (
            self.restaurant_reviews.groupby(["business_id"])
            .agg({"text": list})["text"]
            .values
        )
        self.n_comments_total = sum(
            len(restaurant) for restaurant in self.restaurant_reviews
        )
        self.n_restaurants = len(self.restaurant_reviews)
        self.lens_restaurants = np.array(list(map(len, self.restaurant_reviews)))

        self.batch_size = batch_size
        self.test = test

        self.preprocessor = self.__init_preprocessor(preprocessor)

    def __init_preprocessor(self, preprocessor):
        if not self.test and not preprocessor:
            self.preprocessor = Preprocess(max_features=MAX_FEATURES, maxlen=MAXLEN)
            for i in range(0, self.n_restaurants, 5000):
                high = min(i + 5000, self.n_restaurants)
                all_texts = sum(self.restaurant_reviews[i:high], [])
                self.preprocessor.fit_texts(all_texts)
            print("Fitted the preprocessor on all texts!")
        return preprocessor

    @staticmethod
    def compute_similarity(vec1, vec2):
        return cosine(vec1, vec2)

    def __len__(self):
        return int(np.ceil(self.n_restaurants / self.batch_size))

    def __getitem__(self, idx):
        idx_start = self.batch_size * idx
        idx_end = self.batch_size * (idx + 1)
        x_batch_l = []
        x_batch_r = []
        y_batch = []
        for i in range(idx_start, min(idx_end, self.n_restaurants)):
            n_comments = len(self.restaurant_reviews[i])
            probs = self.lens_restaurants / (self.n_comments_total - n_comments)
            probs[i] = 0

            # 1
            n_positive_examples = np.random.random_integers(
                2, max(3, n_comments // 100)
            )
            positive_examples = np.random.random_integers(
                low=0, high=n_comments - 1, size=(n_positive_examples, 2)
            )
            for ex in positive_examples:
                x_batch_l.append(self.restaurant_reviews[i][ex[0]])
                x_batch_r.append(self.restaurant_reviews[i][ex[1]])
            y_batch.extend(np.ones(n_positive_examples).tolist())
            del positive_examples

            # 0
            n_negative_examples = 3 * np.random.random_integers(
                2, max(3, n_comments // 100)
            )
            negative_restaurants = np.random.choice(
                self.n_restaurants, n_negative_examples, p=probs
            )
            negative_examples = []
            for restaurant in negative_restaurants:
                comment = str(np.random.choice(self.restaurant_reviews[restaurant]))
                # print("COMMENT:\n" + comment + '***************\n')
                negative_examples.append(str(comment))
            del negative_restaurants
            restaurant_comments = self.preprocessor.transform_texts(
                self.restaurant_reviews[i]
            )
            negative_comments = self.preprocessor.transform_texts(negative_examples)

            negative_pairs = []

            for j in range(n_negative_examples):
                comment_0_ind = np.random.randint(0, n_comments)
                negative_pairs.append(
                    (
                        self.compute_similarity(
                            restaurant_comments[comment_0_ind], negative_comments[j]
                        ),
                        comment_0_ind,
                        j,
                    )
                )
            negative_pairs.sort()
            kk = n_negative_examples // 3
            for k in range(kk):
                _, f, s = negative_pairs[k]
                x_batch_l.append(self.restaurant_reviews[i][f])
                x_batch_r.append(negative_examples[s])
            for k in range(2 * kk, n_negative_examples):
                _, f, s = negative_pairs[k]
                x_batch_l.append(self.restaurant_reviews[i][f])
                x_batch_r.append(negative_examples[s])
            y_batch.extend(np.zeros(n_negative_examples - kk).tolist())

            del negative_pairs, negative_examples, restaurant_comments, negative_comments

        x_batch_l, x_batch_r, y_batch = skshuffle(
            x_batch_l, x_batch_r, y_batch, random_state=0
        )

        x_batch_l = self.preprocessor.transform_texts(x_batch_l)
        x_batch_r = self.preprocessor.transform_texts(x_batch_r)
        return [x_batch_l, x_batch_r], y_batch