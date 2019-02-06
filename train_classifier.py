#! /usr/bin/env python
# author: Xinbin Huang - Vancouver School of AI
# date: Dec. 3, 2018
# Usage:
#    python train_classifier.py


import gc
import os
import pickle
import warnings

import numpy as np
import pandas as pd

from bpemb import BPEmb

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from keras.models import Model
from keras.layers import Input, Dense, Embedding, SpatialDropout1D, concatenate
from keras.layers import GRU, Bidirectional, GlobalAveragePooling1D, GlobalMaxPooling1D
from keras.preprocessing import text, sequence
from keras.callbacks import TensorBoard

from utils import get_logger, get_root

np.random.seed(42)
warnings.filterwarnings("ignore")
os.environ["OMP_NUM_THREADS"] = "4"

DIR_ROOT = get_root()
DIR_ASSETS = os.path.join(DIR_ROOT, "assets")
MODEL_PATH = os.path.join(DIR_ASSETS, "model")
LOG_PATH = os.path.join(DIR_ASSETS, "tb_logs")
DATA_FILE = os.path.join(DIR_ROOT, "yelp-data", "rel-train.csv")

MAX_FEATURES = 100000
MAXLEN = 100
EMBED_SIZE = 300
TRAIN_SIZE = 0.90
BATCH_SIZE = 1024
EPOCHS = 2


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


class RocAucEvaluation(TensorBoard):
    def __init__(
        self,
        log_dir="./logs",
        histogram_freq=0,
        batch_size=32,
        write_graph=True,
        write_grads=False,
        write_images=False,
        embeddings_freq=0,
        embeddings_layer_names=None,
        embeddings_metadata=None,
        embeddings_data=None,
        update_freq="epoch",
        validation_data=(),
        interval=1,
    ):
        super().__init__(log_dir=log_dir, batch_size=batch_size)

        self.X_val, self.y_val = validation_data
        self.interval = interval

    def on_epoch_end(self, epoch, logs=None):
        super().on_epoch_end(epoch, logs)

        # if epoch % self.interval == 0:
        #     y_pred = self.model.predict(self.X_val, verbose=0)
        #     score = np.apply_along_axis(roc_auc_score, 0, self.y_val, y_pred)
        #     print(f'\nEpoch: {epoch + 1};\tavg. ROC-AUC: {score.mean():.6f};\tclass-wise ROC-AUC: {score}\n')


def get_model(maxlen, max_features, embed_size, embedding_matrix, class_count):
    input_1 = Input(shape=(maxlen,))
    input_2 = Input(shape=(maxlen,))
    x_1 = Embedding(max_features, embed_size, weights=[embedding_matrix])(input_1)
    x_2 = Embedding(max_features, embed_size, weights=[embedding_matrix])(input_2)
    x = concatenate([x_1, x_2])
    x = SpatialDropout1D(0.2)(x)
    x = Bidirectional(GRU(80, return_sequences=True))(x)
    avg_pool = GlobalAveragePooling1D()(x)
    max_pool = GlobalMaxPooling1D()(x)
    conc = concatenate([avg_pool, max_pool])
    outputs = Dense(class_count, activation="sigmoid")(conc)
    model = Model(inputs=[input_1, input_2], outputs=outputs)
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model


def train():
    logger = get_logger()

    logger.info(f"Loading data: {DATA_FILE}")
    train = pd.read_csv(DATA_FILE)

    print("Started training.")

    features_0 = train["comment_0"].fillna("# #").values
    features_1 = train["comment_1"].fillna("# #").values
    target = train["label"]
    del train
    gc.collect()

    logger.info(f"Transforming data")
    preprocessor = Preprocess(max_features=MAX_FEATURES, maxlen=MAXLEN)
    preprocessor.fit_texts(list(features_0))
    preprocessor.fit_texts(list(features_1))
    features_0 = preprocessor.transform_texts(features_0)
    features_1 = preprocessor.transform_texts(features_1)
    word_index = preprocessor.tokenizer.word_index

    PRERPOCESSOR_FILE = os.path.join(MODEL_PATH, "preprocessor.pkl")
    logger.info(f"Saving the text transformer: {PRERPOCESSOR_FILE}")

    with open(PRERPOCESSOR_FILE, "wb") as file:
        pickle.dump(preprocessor, file)
    del preprocessor
    gc.collect()

    logger.info("Opened preprocessing file.")
    embedding_matrix = get_embeddings(
        word_index, MAX_FEATURES, EMBED_SIZE
    )
    # embedding_matrix = np.zeros((MAX_FEATURES, EMBED_SIZE))  # For quickly testing the validity of the graph

    logger.info(f"Model training, train size: {TRAIN_SIZE}")
    features_0_train, features_0_val, features_1_train, features_1_val, y_train, y_val = train_test_split(
        features_0, features_1, target, train_size=TRAIN_SIZE, random_state=233
    )
    RocAuc = RocAucEvaluation(
        log_dir=LOG_PATH,
        batch_size=BATCH_SIZE,
        validation_data=([features_0_val, features_1_val], y_val),
        interval=1,
    )

    model = get_model(
        MAXLEN, MAX_FEATURES, EMBED_SIZE, embedding_matrix, 1
    )

    logger.info("Model created.")

    hist = model.fit(
        [features_0_train, features_1_train],
        y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=([features_0_val, features_1_val], y_val),
        callbacks=[RocAuc],
        verbose=1,
    )

    ARCHITECTURE_FILE = os.path.join(MODEL_PATH, "gru_architecture.json")
    logger.info(f"Saving the architecture: {ARCHITECTURE_FILE}")

    with open(ARCHITECTURE_FILE, "w") as file:
        architecture_json = model.to_json()
        file.write(architecture_json)

    WEIGHTS_FILE = os.path.join(MODEL_PATH, "gru_weights.h5")
    logger.info(f"Saving the weights: {WEIGHTS_FILE}")

    model.save_weights(WEIGHTS_FILE)

    logger.info("Model saved.")


if __name__ == "__main__":
    train()