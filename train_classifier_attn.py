import gc
import os
import pickle
import warnings

import numpy as np
import pandas as pd

from bpemb import BPEmb

from sklearn.model_selection import train_test_split

from keras.preprocessing import text, sequence
from keras.models import Model
from keras.layers import Dense, Input, LSTM, Embedding, Dropout
from keras.layers.merge import concatenate
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, TensorBoard

from Attention import Attention

from utils import get_logger, get_root

from yelp_dataset_generator import *


np.random.seed(42)
warnings.filterwarnings("ignore")
os.environ["OMP_NUM_THREADS"] = "4"

DIR_ROOT = get_root()
DIR_ASSETS = os.path.join(DIR_ROOT, "assets")
MODEL_PATH = os.path.join(DIR_ASSETS, "model")
LOG_PATH = os.path.join(DIR_ASSETS, "tb_logs")

MAX_SEQUENCE_LENGTH = 150
MAX_NB_WORDS = 100000
EMBEDDING_DIM = 300
DENSE_SIZE = 256
LSTM_SIZE = 300
RATE_DROP_LSTM = 0.25
RATE_DROP_DENSE = 0.25
VALIDATION_SPLIT = 0.1
TRAIN_SIZE = 1 - VALIDATION_SPLIT
MAX_FEATURES = 100000
BATCH_SIZE = 1024
EPOCHS = 1
act = "relu"


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


def get_model(maxlen, max_features, lstm_size, rate_drop_lstm, rate_drop_dense, embed_size, embedding_matrix):
    input_1 = Input(shape=(maxlen,))
    input_2 = Input(shape=(maxlen,))
    emb_layer = Embedding(max_features, embed_size, weights=[embedding_matrix])
    embedding_layer_1 = emb_layer(input_1)
    embedding_layer_2 = emb_layer(input_2)
    embedded_sequences = concatenate([embedding_layer_1, embedding_layer_2])
    x = LSTM(lstm_size,
             dropout=rate_drop_lstm,
             recurrent_dropout=rate_drop_lstm,
             return_sequences=True)(embedded_sequences)
    x = Dropout(rate_drop_dense)(x)
    merged = Attention()(x)
    merged = Dense(DENSE_SIZE, activation=act)(merged)
    merged = Dropout(rate_drop_dense)(merged)
    # merged = BatchNormalization()(merged)
    outputs = Dense(1, activation='sigmoid')(merged)

    model = Model(inputs=[input_1, input_2], outputs=outputs)
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    model.summary()

    return model


def train():
    logger = get_logger()
    logger.info(f"Transforming data")

    PRERPOCESSOR_FILE = os.path.join(MODEL_PATH, "preprocessor_attn.pkl")
    with open(PRERPOCESSOR_FILE, 'rb') as f:
        preprocesor = pickle.load(f)
        yelp_dataset_generator = YELPSequence(batch_size=16, test=False, preproc=preprocesor)
        logger.info("Opened preprocessing file.")
    if yelp_dataset_generator is None:
        YELPSequence(batch_size=16, test=False)

    logger.info(f"Saving the text transformer: {PRERPOCESSOR_FILE}")
    with open(PRERPOCESSOR_FILE, "wb") as file:
        pickle.dump(yelp_dataset_generator.preprocessor, file)

    word_index = yelp_dataset_generator.preprocessor.tokenizer.word_index
    embedding_matrix = get_embeddings(word_index, MAX_FEATURES, EMBEDDING_DIM)

    yelp_dataset_generator_val = YELPSequence(batch_size=16, test=True)

    logger.info(f"Model training, train size: {TRAIN_SIZE}")
    """
    RocAuc = RocAucEvaluation(
        log_dir=LOG_PATH,
        batch_size=BATCH_SIZE,
        interval=1,
    )
    """
    model = get_model(MAXLEN, MAX_FEATURES, LSTM_SIZE, RATE_DROP_LSTM, RATE_DROP_DENSE, EMBEDDING_DIM, embedding_matrix)

    logger.info("Model created.")

    hist = model.fit_generator(yelp_dataset_generator, steps_per_epoch=None, epochs=1, verbose=1, callbacks=None,
                               validation_data=yelp_dataset_generator_val,
                               validation_steps=None, class_weight=None, max_queue_size=1000,
                               workers=16, use_multiprocessing=True, shuffle=True, initial_epoch=0)

    early_stopping = EarlyStopping(monitor='val_loss', patience=5)

    ARCHITECTURE_FILE = os.path.join(MODEL_PATH, "gru_architecture_attn.json")
    logger.info(f"Saving the architecture: {ARCHITECTURE_FILE}")

    with open(ARCHITECTURE_FILE, "w") as file:
        architecture_json = model.to_json()
        file.write(architecture_json)

    WEIGHTS_FILE = os.path.join(MODEL_PATH, "gru_weights_attn.h5")
    logger.info(f"Saving the weights: {WEIGHTS_FILE}")

    model.save_weights(WEIGHTS_FILE)


if __name__ == "__main__":
    train()
