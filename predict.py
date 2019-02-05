#! /usr/bin/env python
# author: Xinbin Huang - Vancouver School of AI
# date: Dec. 3, 2018

import os

from train_classifier_attn import *

from utils import get_root, load_pipeline_stages
from train_classifier import Preprocess  # for unpickling to work properly


ROOT = get_root()
MODEL_PATH = os.path.join(ROOT, "assets", "model")
PREPROCESSOR_FILE = os.path.join(MODEL_PATH, "preprocessor.pkl")
ARCHITECTURE_FILE = os.path.join(MODEL_PATH, "gru_architecture.json")
WEIGHTS_FILE = os.path.join(MODEL_PATH, "gru_weights.h5")


class PredictionPipeline(object):
    def __init__(self, preprocessor, model):
        self.preprocessor = preprocessor
        self.model = model

    def predict(self, sent):
        features_0 = sent[0]
        features_1 = sent[1]
        features_0 = self.preprocessor.transform_texts(features_0)
        features_1 = self.preprocessor.transform_texts(features_1)

        return self.model.predict([features_0, features_1])


def load_pipeline():
    return PredictionPipeline(
        *load_pipeline_stages(PREPROCESSOR_FILE, ARCHITECTURE_FILE, WEIGHTS_FILE)
    )


if __name__ == "__main__":
    ppl = load_pipeline()

    sample_text = [["nice food, mc Donald's is the best", "Beef was very tasty", "Awful car service!"],
                   ["mc Donald's is the best restaurant I've ever seen!", "It's a great restaurant for vegans!", "Go go Arsenal!"]]

    print(f"Relevance: {ppl.predict(sample_text)}")

