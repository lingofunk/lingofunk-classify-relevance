import gc
import os
import pickle
import warnings

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from keras.preprocessing import text, sequence
from keras.models import Model
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation
from keras.layers.merge import concatenate
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard

from keras import backend as K
from keras.engine.topology import Layer
from keras import initializers, regularizers, constraints

from utils import get_logger, get_root


np.random.seed(42)
warnings.filterwarnings("ignore")
os.environ["OMP_NUM_THREADS"] = "4"

DIR_ROOT = get_root()
DIR_ASSETS = os.path.join(DIR_ROOT, "assets")
MODEL_PATH = os.path.join(DIR_ASSETS, "model")
LOG_PATH = os.path.join(DIR_ASSETS, "tb_logs")
EMBEDDING_FILE = os.path.join(
    DIR_ASSETS, "embedding", "glove.840B.300d.txt"
)
DATA_FILE = os.path.join(DIR_ASSETS, "data", "train.csv")

MAX_SEQUENCE_LENGTH = 150
MAX_NB_WORDS = 100000
EMBEDDING_DIM = 300
DENSE_SIZE = 256
LSTM_SIZE = 300
RATE_DROP_LSTM = 0.25
RATE_DROP_DENSE = 0.25
VALIDATION_SPLIT = 0.1
TRAIN_SIZE = 1 - VALIDATION_SPLIT
MAX_FEATURES = 30000
BATCH_SIZE = 128
EPOCHS = 2
act = "relu"


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


def get_embeddings(embed_file, word_index, max_features, embed_size):
    embeddings_pretrained = {}
    i = 0
    f = open(embed_file)
    for line in f:
        if i % 500000 == 0:
            print(i)
        values = line.split()
        try:
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_pretrained[word] = coefs
            i += 1
        except ValueError:
            continue
    f.close()


    def get_coefs(word, *arr):
        return word, np.asarray(arr, dtype="float32")

    nb_words = min(max_features, len(word_index))
    embedding_matrix = np.zeros((nb_words, embed_size))
    for word, i in word_index.items():
        if i >= max_features:
            continue
        embedding_vector = embeddings_pretrained.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    del embeddings_pretrained

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


class Attention(Layer):
    def __init__(self, step_dim,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        """
        Keras Layer that implements an Attention mechanism for temporal data.
        Supports Masking.
        Follows the work of Raffel et al. [https://arxiv.org/abs/1512.08756]
        # Input shape
            3D tensor with shape: `(samples, steps, features)`.
        # Output shape
            2D tensor with shape: `(samples, features)`.
        :param kwargs:
        Just put it on top of an RNN Layer (GRU/LSTM/SimpleRNN) with return_sequences=True.
        The dimensions are inferred based on the output shape of the RNN.
        Example:
            model.add(LSTM(64, return_sequences=True))
            model.add(Attention())
        """
        self.supports_masking = True
        # self.init = initializations.get('glorot_uniform')
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.step_dim = step_dim
        self.features_dim = 0
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight((input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        self.features_dim = input_shape[-1]

        if self.bias:
            self.b = self.add_weight((input_shape[1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None

        self.built = True

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):
        # eij = K.dot(x, self.W) TF backend doesn't support it

        # features_dim = self.W.shape[0]
        # step_dim = x._keras_shape[1]

        features_dim = self.features_dim
        step_dim = self.step_dim

        eij = K.reshape(K.dot(K.reshape(x, (-1, features_dim)), K.reshape(self.W, (features_dim, 1))), (-1, step_dim))

        if self.bias:
            eij += self.b

        eij = K.tanh(eij)

        a = K.exp(eij)

        # apply mask after the exp. will be re-normalized next
        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            a *= K.cast(mask, K.floatx())

        # in some cases especially in the early stages of training the sum may be almost zero
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a
        # print weigthted_input.shape
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        # return input_shape[0], input_shape[-1]
        return input_shape[0], self.features_dim


def get_model(maxlen, max_features, lstm_size, rate_drop_lstm, rate_drop_dense, embed_size, embedding_matrix):
    input_1 = Input(shape=(maxlen,))
    input_2 = Input(shape=(maxlen,))
    embedding_layer_1 = Embedding(max_features, embed_size, weights=[embedding_matrix])(input_1)
    embedding_layer_2 = Embedding(max_features, embed_size, weights=[embedding_matrix])(input_2)
    embedded_sequences = concatenate([embedding_layer_1, embedding_layer_2])
    x = LSTM(lstm_size,
             dropout=rate_drop_lstm,
             recurrent_dropout=rate_drop_lstm,
             return_sequences=True)(embedded_sequences)
    x = Dropout(rate_drop_dense)(x)
    merged = Attention(maxlen)(x)
    merged = Dense(DENSE_SIZE, activation=act)(merged)
    merged = Dropout(rate_drop_dense)(merged)
    merged = BatchNormalization()(merged)
    outputs = Dense(1, activation='sigmoid')(merged)

    model = Model(inputs=[input_1, input_2], outputs=outputs)
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

    return model


def train():
    logger = get_logger()

    logger.info(f"Loading data: {DATA_FILE}")
    train = pd.read_csv(DATA_FILE)

    features_0 = train["comment_0"].fillna("# #").values
    features_1 = train["comment_1"].fillna("# #").values
    # target = convert_binary_toxic(train, config.classes)
    # target = train[config.classes]
    target = train["label"]
    del train
    gc.collect()

    logger.info(f"Transforming data")
    preprocessor = Preprocess(max_features=MAX_FEATURES, maxlen=MAX_SEQUENCE_LENGTH)
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

    logger.info(f"Loading embedding vectors: {EMBEDDING_FILE}")
    embedding_matrix = get_embeddings(
        EMBEDDING_FILE, word_index, MAX_FEATURES, EMBEDDING_DIM
    )
    # embedding_matrix = np.zeros((MAX_FEATURES, EMBED_SIZE))  # For quickly testing the validity of the graph

    logger.info(f"Model training, train size: {TRAIN_SIZE}")
    X_train, X_val, y_train, y_val = train_test_split(
        [features_0, features_1], target, train_size=TRAIN_SIZE, random_state=233
    )
    RocAuc = RocAucEvaluation(
        log_dir=LOG_PATH,
        batch_size=BATCH_SIZE,
        validation_data=(X_val, y_val),
        interval=1,
    )

    early_stopping = EarlyStopping(monitor='val_loss', patience=5)

    model = get_model(
        MAX_SEQUENCE_LENGTH, MAX_FEATURES, LSTM_SIZE, RATE_DROP_LSTM, RATE_DROP_DENSE, EMBEDDING_DIM, embedding_matrix)

    hist = model.fit(
        X_train,
        y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(X_val, y_val),
        callbacks=[RocAuc, early_stopping],
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


if __name__ == "__main__":
    train()
