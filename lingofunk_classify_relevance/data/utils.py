import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from bpemb import BPEmb
from keras.models import model_from_json

from lingofunk_classify_relevance.config import fetch_constant, fetch_data
from lingofunk_classify_relevance.model.layers.attention import Attention

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


def get_root():
    """Return project root folder"""
    return Path(__file__).parent.parent


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


def get_logger(level=logging.INFO):
    """A simple logger for tracking training process"""
    logger = logging.getLogger(__name__)
    logger.setLevel(level)

    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    handler.setFormatter(formatter)
    logger.addHandler(handler)

    logger.info("Logging successfully configured!")

    return logger


def load_model(architecture_file, weights_file):
    with open(architecture_file) as arch_json:
        architecture = arch_json.read()
    model = model_from_json(architecture)
    model.load_weights(weights_file)
    return model


def load_preprocessor(preprocessor_file, logger=get_logger()):
    VOCAB_MAX_FEATURES = fetch_constant("VOCAB_MAX_FEATURES")
    WORD_MAX_LEN = fetch_constant("WORD_MAX_LEN")
    PATH_TO_YELP_CSV_TRAIN = fetch_data("train")

    try:
        with open(preprocessor_file, "rb") as f:
            preprocessor = pickle.load(f)
            logger.info("Opened preprocessing file.")
            return preprocessor
    except FileNotFoundError:
        text = (
            pd.read_csv(PATH_TO_YELP_CSV_TRAIN)
            .groupby(["business_id"])
            .agg({"text": list})["text"]
            .values
        )
        preprocessor = Preprocess(max_features=VOCAB_MAX_FEATURES, maxlen=WORD_MAX_LEN)
        for i in range(0, len(text), 5000):
            high = min(i + 5000, len(text))
            all_texts = sum(text[i:high], [])
            preprocessor.fit_texts(all_texts)
        print("Fitted the preprocessor on all texts!")

        with open(preprocessor_file, "wb") as file:
            pickle.dump(preprocessor, file)

        logger.info(f"Saving the text transformer: {preprocessor_file}")

        return preprocessor


def load_pipeline_stages(preprocessor_file, architecture_file, weights_file):
    preprocessor = load_preprocessor(preprocessor_file)

    json_file = open(architecture_file, "r")
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(
        loaded_model_json, custom_objects={"Attention": Attention}
    )
    loaded_model.load_weights(weights_file)
    print("Loaded Model from disk")
    return preprocessor, loaded_model
