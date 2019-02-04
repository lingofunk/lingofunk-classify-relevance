#! /usr/bin/env python
# author: Xinbin Huang - Vancouver School of AI
# date: Dec. 3, 2018

import os

from .utils import get_root, load_pipeline_stages
from .train_classifier import Preprocess  # for unpickling to work properly


ROOT = get_root()
MODEL_PATH = os.path.join(ROOT, "assets", "model")
PREPROCESSOR_FILE = os.path.join(MODEL_PATH, "preprocessor.pkl")
ARCHITECTURE_FILE = os.path.join(MODEL_PATH, "gru_architecture.json")
WEIGHTS_FILE = os.path.join(MODEL_PATH, "gru_weights.h5")


class PredictionPipeline(object):
    def __init__(self, preprocessor, model):
        self.preprocessor = preprocessor
        self.model = model

    def get_features(self, text):
        return self.preprocessor.transform_texts(text)

    def predict(self, text):
        features = self.get_features(text)
        return self.model.predict(features)


def load_pipeline():
    return PredictionPipeline(
        *load_pipeline_stages(PREPROCESSOR_FILE, ARCHITECTURE_FILE, WEIGHTS_FILE)
    )


if __name__ == "__main__":
    ppl = load_pipeline()

    sample_text = [["nice food, mc Donald's is the best", "mc Donald's is the best restaurant I've ever seen!"],
                   ["Beef was very tasty", "It's a great restaurant for vegans!"],
                   ["Awful car service!", "Go go Arsenal!"]]

    for text, toxicity in zip(sample_text, ppl.predict(sample_text)):
        print(f"{text}".ljust(25) + f"- Toxicity: {toxicity}")
