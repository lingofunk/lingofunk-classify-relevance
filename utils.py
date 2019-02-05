#! /usr/bin/env python

import logging
from pathlib import Path
import pickle

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
    from . import train_classifier
    from .train_classifier import Preprocess  # for unpickling to work properly
    preprocessor = pickle.load(open(preprocessor_file, 'rb'))
    model = load_model(architecture_file, weights_file)
    return preprocessor, model
