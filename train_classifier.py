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

from sklearn.model_selection import train_test_split

from keras.models import Model
from keras.layers import Input, Dense, Embedding, SpatialDropout1D, concatenate
from keras.layers import GRU, Bidirectional, GlobalAveragePooling1D, GlobalMaxPooling1D
from keras.callbacks import TensorBoard

from utils import get_logger, get_root

from yelp_dataset_generator import *

np.random.seed(42)
warnings.filterwarnings("ignore")
os.environ["OMP_NUM_THREADS"] = "4"

DIR_ROOT = get_root()
DIR_ASSETS = os.path.join(DIR_ROOT, "assets")
MODEL_PATH = os.path.join(DIR_ASSETS, "model")
LOG_PATH = os.path.join(DIR_ASSETS, "tb_logs")
# DATA_FILE = os.path.join(DIR_ROOT, "yelp-data", "rel-train.csv")

MAX_FEATURES = 100000
MAXLEN = 100
EMBED_SIZE = 300
TRAIN_SIZE = 0.90
BATCH_SIZE = 1024
EPOCHS = 2

"""
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

"""


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

    logger.info(f"Transforming data")

    yelp_dataset_generator = YELPSequence(batch_size=128)

    PRERPOCESSOR_FILE = os.path.join(MODEL_PATH, "preprocessor.pkl")

    try:
        print("PREPROC")
        with open(PRERPOCESSOR_FILE, 'rb') as f:
            preprocesor = pickle.load(f)
            print("PREPROC_____")
            yelp_dataset_generator.preprocess(preprocesor)

            logger.info("Opened preprocessing file.")
    except:
        yelp_dataset_generator.preprocess()

    logger.info(f"Saving the text transformer: {PRERPOCESSOR_FILE}")
    with open(PRERPOCESSOR_FILE, "wb") as file:
        pickle.dump(yelp_dataset_generator.preprocessor, file)

    word_index = yelp_dataset_generator.preprocessor.tokenizer.word_index
    embedding_matrix = get_embeddings(word_index, MAX_FEATURES, EMBED_SIZE)

    logger.info(f"Model training, train size: {TRAIN_SIZE}")
    """
    RocAuc = RocAucEvaluation(
        log_dir=LOG_PATH,
        batch_size=BATCH_SIZE,
        interval=1,
    )
    """
    model = get_model(
        MAXLEN, MAX_FEATURES, EMBED_SIZE, embedding_matrix, 1
    )

    logger.info("Model created.")

    hist = model.fit_generator(yelp_dataset_generator, steps_per_epoch=None, epochs=1, verbose=1, callbacks=None,
                               validation_data=None, validation_steps=None, class_weight=None, max_queue_size=10,
                               workers=4, use_multiprocessing=False, shuffle=True, initial_epoch=0)

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
