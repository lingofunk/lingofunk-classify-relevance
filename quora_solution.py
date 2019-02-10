from __future__ import print_function
import numpy as np
import csv, datetime, time, json
from zipfile import ZipFile
from os.path import expanduser, exists
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.layers import Input, TimeDistributed, Dense, Lambda, concatenate, Dropout, BatchNormalization
from keras.layers.embeddings import Embedding
from keras.callbacks import Callback, ModelCheckpoint
from keras.utils.data_utils import get_file
from keras import backend as K
from sklearn.model_selection import train_test_split
import warnings
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from keras.models import Model
from keras.layers import Dense, Input, Bidirectional, LSTM, Embedding, Dropout, Add
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint


from yelp_dataset_generator import *

np.random.seed(42)
warnings.filterwarnings("ignore")
os.environ["OMP_NUM_THREADS"] = "4"

# Initialize global variables
MAX_FEATURES = 100000
MAX_SEQUENCE_LENGTH = 150
EMBEDDING_DIM = 300
DENSE_SIZE = 200
DROPOUT = 0.1
BATCH_SIZE = 128
OPTIMIZER = 'adam'

PREPROCESSOR_FILE = os.path.join(MODEL_PATH, "preprocessor_attn.pkl")
ARCHITECTURE_FILE = os.path.join(MODEL_PATH, "architecture_quora.json")
WEIGHTS_FILE = os.path.join(MODEL_PATH, "weights_quora.h5")


def get_model(maxlen, max_features, dropout, dense_size, embed_size, embedding_matrix):
    input_1 = Input(shape=(maxlen,))
    input_2 = Input(shape=(maxlen,))
    emb_layer = Embedding(max_features, embed_size, weights=[embedding_matrix], trainable=False)
    embedding_layer_1 = emb_layer(input_1)
    embedding_layer_2 = emb_layer(input_2)

    q1 = TimeDistributed(Dense(embed_size, activation='relu'))(embedding_layer_1)
    q1 = Lambda(lambda x: K.max(x, axis=1), output_shape=(embed_size,))(q1)
    q2 = TimeDistributed(Dense(embed_size, activation='relu'))(embedding_layer_2)
    q2 = Lambda(lambda x: K.max(x, axis=1), output_shape=(embed_size,))(q2)

    merged = concatenate([q1, q2])
    merged = Dense(dense_size, activation='relu')(merged)
    merged = Dropout(dropout)(merged)
    merged = BatchNormalization()(merged)
    merged = Dense(dense_size, activation='relu')(merged)
    merged = Dropout(dropout)(merged)
    merged = BatchNormalization()(merged)
    merged = Dense(dense_size, activation='relu')(merged)
    merged = Dropout(dropout)(merged)
    merged = BatchNormalization()(merged)
    merged = Dense(dense_size, activation='relu')(merged)
    merged = Dropout(dropout)(merged)
    merged = BatchNormalization()(merged)

    outputs = Dense(1, activation='sigmoid')(merged)

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

    try:
        with open(PREPROCESSOR_FILE, 'rb') as f:
            preprocessor = pickle.load(f)
            yelp_dataset_generator = YELPSequence(batch_size=BATCH_SIZE, test=False, preproc=preprocessor)
            logger.info("Opened preprocessing file.")
    except FileNotFoundError:
        yelp_dataset_generator = YELPSequence(batch_size=BATCH_SIZE, test=False)

    logger.info(f"Saving the text transformer: {PREPROCESSOR_FILE}")

    with open(PREPROCESSOR_FILE, "wb") as file:
        pickle.dump(yelp_dataset_generator.preprocessor, file)

    word_index = yelp_dataset_generator.preprocessor.tokenizer.word_index
    embedding_matrix = get_embeddings(word_index, MAX_FEATURES, EMBEDDING_DIM)

    yelp_dataset_generator_val = YELPSequence(batch_size=BATCH_SIZE,
                                              test=True, preproc=yelp_dataset_generator.preprocessor)

    model = get_model(MAXLEN, MAX_FEATURES, DROPOUT, DENSE_SIZE, EMBEDDING_DIM, embedding_matrix)

    logger.info("Model created.")

    early_stopping = EarlyStopping(monitor='val_loss', patience=2)
    model_checkpoint = ModelCheckpoint(os.path.join(MODEL_PATH, "model_quora.h5"), monitor='val_loss',
                                       verbose=1, save_best_only=True,
                                       save_weights_only=False, mode='auto', period=1)

    wanna_train = True
    n_epochs = 250

    while wanna_train:
        hist = model.fit_generator(yelp_dataset_generator, steps_per_epoch=None, epochs=n_epochs, verbose=1,
                                   callbacks=[early_stopping, model_checkpoint],
                                   validation_data=yelp_dataset_generator_val,
                                   validation_steps=100, class_weight=None, max_queue_size=10000,
                                   workers=16, use_multiprocessing=True, shuffle=True, initial_epoch=0)

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
