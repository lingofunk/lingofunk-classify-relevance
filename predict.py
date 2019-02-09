#! /usr/bin/env python
# author: Xinbin Huang - Vancouver School of AI
# date: Dec. 3, 2018

import os

from train_classifier_attn import *

from utils import get_root, load_pipeline_stages

ROOT = get_root()
MODEL_PATH = os.path.join(ROOT, "yelp-data", "model")
PREPROCESSOR_FILE = os.path.join(MODEL_PATH, "preprocessor_attn.pkl")
ARCHITECTURE_FILE = os.path.join(MODEL_PATH, "gru_architecture_attn.json")
WEIGHTS_FILE = os.path.join(MODEL_PATH, "gru_weights_attn.h5")


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


if __name__ == "__main__":
    ppl = load_pipeline()

    sample_text = [["nice food, mc Donald's is the best", "Beef was very tasty", "Awful car service!"],
                   ["mc Donald's is the best restaurant I've ever seen!", "It's a great restaurant for vegans!", "Go go Arsenal!"]]

    print(f"Relevance: {ppl.predict(sample_text)}")
    while True:
        print("Enter two comments.")
        s1 = input()
        s2 = input()
        print(f"Relevance: {ppl.predict([[s1], [s2]])}")
