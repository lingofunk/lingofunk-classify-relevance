#! /usr/bin/env python
import os

import tensorflow as tf

from lingofunk_classify_relevance.config import fetch_model
from lingofunk_classify_relevance.data.utils import load_pipeline_stages

PREPROCESSOR_FILE = fetch_model("current", "preprocessor")
ARCHITECTURE_FILE = fetch_model("current", "architecture")
WEIGHTS_FILE = fetch_model("current", "weights")


class PredictionPipeline(object):
    def __init__(self, preprocessor, model):
        self.preprocessor = preprocessor
        self.model = model
        self.graph = tf.get_default_graph()

    def predict(self, sent):
        with self.graph.as_default():
            features_0 = self.preprocessor.transform_texts(sent[0])
            features_1 = self.preprocessor.transform_texts(sent[1])

            return self.model.predict([features_0, features_1])


def load_pipeline():
    return PredictionPipeline(
        *load_pipeline_stages(PREPROCESSOR_FILE, ARCHITECTURE_FILE, WEIGHTS_FILE)
    )


class ReviewComparer:
    def __init__(self):
        self.ppl = load_pipeline()

    def answer_query(self, review1: str, review2: str):
        return self.ppl.predict([[review1], [review2]])[0][0]

    def answer_queries(self, reviews1: list, reviews2: list):
        return list(map(lambda x: x[0], self.ppl.predict([reviews1, reviews2])))


if __name__ == "__main__":
    ppl = load_pipeline()

    sample_text = [
        [
            "nice food, mc Donald's is the best",
            "Beef was very tasty",
            "Awful car service!",
        ],
        [
            "mc Donald's is the best restaurant I've ever seen!",
            "It's a great restaurant for vegans!",
            "Go go Arsenal!",
        ],
    ]

    print(f"Relevance: {ppl.predict(sample_text)}")
    while True:
        print("Enter two comments.")
        s1 = input()
        s2 = input()
        print(f"Relevance: {ppl.predict([[s1], [s2]])}")
