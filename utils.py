import os
import logging
from pathlib import Path
import pickle
from Attention import Attention
from keras.models import model_from_json


def get_root():
    """Return project root folder"""
    return Path(__file__).parent.parent


def get_logger(level=logging.INFO):
    """A simple logger for tracking training process"""
    logger = logging.getLogger(__name__)
    logger.setLevel(level)

    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

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


def load_pipeline_stages(preprocessor_file, architecture_file, weights_file):
    # from train_classifier import Preprocess  # for unpickling to work properly
    preprocessor = pickle.load(open(preprocessor_file, 'rb'))

    json_file = open(architecture_file, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json, custom_objects={'Attention': Attention})
    loaded_model.load_weights(weights_file)
    print("Loaded Model from disk")
    return preprocessor, loaded_model


DIR_ROOT = get_root()
DIR_ASSETS = os.path.join(DIR_ROOT, "assets")
DATA_DIR = os.path.join(DIR_ROOT, "yelp-data")
MODEL_PATH = os.path.join(DIR_ASSETS, "model")
LOG_PATH = os.path.join(DIR_ASSETS, "tb_logs")
PATH_TO_YELP_CSV = os.path.join(DATA_DIR, "restaurant_reviews.csv")
PATH_TO_YELP_CSV_TRAIN = os.path.join(DATA_DIR, "reviews_train.csv")
PATH_TO_YELP_CSV_TEST = os.path.join(DATA_DIR, "reviews_test.csv")
