import os
import warnings

import numpy as np
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import LSTM, Add, Bidirectional, Dense, Dropout, Embedding, Input
from keras.layers.normalization import BatchNormalization
from keras.models import Model

from lingofunk_classify_relevance.config import fetch_model
from lingofunk_classify_relevance.data.utils import get_embeddings, get_logger
from lingofunk_classify_relevance.data.data_generator import YELPSequence
from lingofunk_classify_relevance.model.layers.attention import Attention

np.random.seed(42)
warnings.filterwarnings("ignore")
os.environ["OMP_NUM_THREADS"] = "4"

PREPROCESSOR_FILE = fetch_model("attnet", "preprocessor")
ARCHITECTURE_FILE = fetch_model("attnet", "architecture")
WEIGHTS_FILE = fetch_model("attnet", "weights")

MAX_SEQUENCE_LENGTH = 150
MAX_NB_WORDS = 100_000
EMBEDDING_DIM = 300
DENSE_SIZE = 256
LSTM_SIZE = 300
RATE_DROP_LSTM = 0.25
RATE_DROP_DENSE = 0.25
MAX_FEATURES = 100_000
BATCH_SIZE = 128
EPOCHS = 1
act = "relu"


def get_model(
    maxlen,
    max_features,
    lstm_size,
    rate_drop_lstm,
    rate_drop_dense,
    embed_size,
    embedding_matrix,
):
    input_1 = Input(shape=(maxlen,))
    input_2 = Input(shape=(maxlen,))
    emb_layer = Embedding(
        max_features, embed_size, weights=[embedding_matrix], trainable=False
    )
    embedding_layer_1 = emb_layer(input_1)
    embedding_layer_2 = emb_layer(input_2)
    # embedded_sequences = concatenate([embedding_layer_1, embedding_layer_2], axis=-1)
    # embedded_sequences = Add()([embedding_layer_1, embedding_layer_2])
    embedded_sequences = Add()([embedding_layer_1, embedding_layer_2])
    x = Bidirectional(
        LSTM(
            lstm_size,
            dropout=rate_drop_lstm,
            recurrent_dropout=rate_drop_lstm,
            return_sequences=True,
        )
    )(embedded_sequences)
    x = Dropout(rate_drop_dense)(x)
    merged = Attention()(x)
    merged = Dense(DENSE_SIZE, activation=act)(merged)
    merged = Dropout(rate_drop_dense)(merged)
    merged = BatchNormalization()(merged)
    outputs = Dense(1, activation="sigmoid")(merged)

    model = Model(inputs=[input_1, input_2], outputs=outputs)
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    model.summary()

    return model


def train():
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.5
    set_session(tf.Session(config=config))

    logger = get_logger()
    logger.info(f"Transforming data")

    word_index = data_generator.preprocessor.tokenizer.word_index
    embedding_matrix = get_embeddings(word_index, MAX_FEATURES, EMBEDDING_DIM)

    data_generator_val = YELPSequence(
        batch_size=BATCH_SIZE,
        test=True,
        preprocessor=data_generator.preprocessor,
    )

    model = get_model(
        MAXLEN,
        MAX_FEATURES,
        LSTM_SIZE,
        RATE_DROP_LSTM,
        RATE_DROP_DENSE,
        EMBEDDING_DIM,
        embedding_matrix,
    )

    logger.info("Model created.")

    early_stopping = EarlyStopping(monitor="val_loss", patience=5)
    model_checkpoint = ModelCheckpoint(
        WEIGHTS_FILE,
        monitor="val_loss",
        verbose=1,
        save_best_only=True,
        save_weights_only=False,
        mode="auto",
        period=1,
    )

    wanna_train = True
    n_epochs = 250

    while wanna_train:
        hist = model.fit_generator(
            data_generator,
            steps_per_epoch=None,
            epochs=n_epochs,
            verbose=1,
            callbacks=[early_stopping, model_checkpoint],
            validation_data=data_generator_val,
            validation_steps=100,
            class_weight=None,
            max_queue_size=10000,
            workers=16,
            use_multiprocessing=True,
            shuffle=True,
            initial_epoch=0,
        )

        logger.info(f"Saving the architecture: {ARCHITECTURE_FILE}")

        with open(ARCHITECTURE_FILE, "w") as file:
            architecture_json = model.to_json()
            file.write(architecture_json)

        logger.info(f"Saving the weights: {WEIGHTS_FILE}")
        model.save_weights(WEIGHTS_FILE)

        ans = int(input("How many epochs more?"))
        if ans == 0:
            wanna_train = False
        else:
            n_epochs = ans


if __name__ == "__main__":
    train()
