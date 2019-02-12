import pickle
import time
from itertools import product

import numpy as np
import pandas as pd

from lingofunk_classify_relevance.config import fetch_data
from lingofunk_classify_relevance.predict import ReviewComparer

DATASET_CSV = fetch_data("city")
SIMILARITY_MATRIX = fetch_data("citymatrix")


class TownTextExtractor:
    def __init__(self, similarity_matrix=None):
        self.restaurant_reviews = pd.read_csv(DATASET_CSV)

        self.restaurant_reviews = self.restaurant_reviews.groupby(["business_id"]).agg(
            {"business_id": tuple, "text": list}
        )

        self.id2rest = list(
            map(lambda x: x[0], self.restaurant_reviews["business_id"].values)
        )
        self.rest2id = dict()
        for i, rest in enumerate(self.id2rest):
            self.rest2id[rest] = i

        self.restaurant_reviews = self.restaurant_reviews["text"].values
        self.n_comments_total = sum(
            len(restaurant) for restaurant in self.restaurant_reviews
        )
        self.n_restaurants = len(self.restaurant_reviews)
        self.lens_restaurants = np.array(list(map(len, self.restaurant_reviews)))

        self.comparer = ReviewComparer()

        if similarity_matrix is not None:
            self.similarity_matrix = similarity_matrix
        else:
            self.similarity_matrix = np.zeros(
                shape=(self.n_restaurants, self.n_restaurants)
            )
        self.uniqueness = np.zeros(shape=(self.n_restaurants,))
        self.uniqueness_rest = np.zeros(shape=(self.n_restaurants,))
        self.uniqueness_ids = list()
        self.uniqueness_sorted = np.zeros(shape=(self.n_restaurants,))
        self.n_total = 0

    def compute_similarity_matrix(self):
        total_time = 0
        for i in range(self.n_restaurants):
            restaurants_i = self.restaurant_reviews[i]
            for j in range(i + 1, self.n_restaurants):
                start = time.time()
                n_ij = self.lens_restaurants[i] * self.lens_restaurants[j]
                restaurants_j = self.restaurant_reviews[j]
                queries_i = [r_i for r_i, r_j in product(restaurants_i, restaurants_j)]
                queries_j = [r_j for r_i, r_j in product(restaurants_i, restaurants_j)]

                ans_ij = self.comparer.answer_queries(queries_i, queries_j)
                self.similarity_matrix[i][j] = np.mean(ans_ij)
                self.similarity_matrix[j][i] = self.similarity_matrix[i][j]
                finish = time.time()
                self.n_total += n_ij
                print(i, j, "\t\t", n_ij, finish - start)
                total_time += finish - start
        out = open(SIMILARITY_MATRIX, "wb")
        pickle.dump(self.similarity_matrix, out)
        print("TIME: ", total_time)

    def load_similarity_matrix(self):
        inp = open(SIMILARITY_MATRIX, "rb")
        self.similarity_matrix = pickle.load(inp)
        self.uniqueness = np.sum(self.similarity_matrix, axis=-1)
        self.uniqueness_ids = np.argsort(self.uniqueness)[::-1]
        self.uniqueness_rest = list(map(lambda x: self.id2rest[x], self.uniqueness_ids))
        self.uniqueness_sorted = np.sort(self.uniqueness)[::-1]

    def get_heatmap_for_restaurant_id(self, i):
        return self.similarity_matrix[i] / sum(self.similarity_matrix[i])

    def get_heatmap_for_restaurant_name(self, rest):
        return self.get_heatmap_for_restaurant_id(self.rest2id[rest])

    def get_heatmap_for_restaurant(self, q):
        return list(
            sorted(
                zip(self.id2rest, self.get_heatmap_for_restaurant_name(q)),
                key=lambda x: -x[1],
            )
        )

    def get_unique_restaurants(self):
        return list(zip(self.uniqueness_rest, self.uniqueness_sorted))
