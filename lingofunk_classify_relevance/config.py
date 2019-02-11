import codecs
import logging
import operator
import os
from functools import reduce
from pathlib import Path

from poyo import PoyoException, parse_string

logging.basicConfig(level=logging.DEBUG)

ROOT = Path(__file__).parent
CONFIG_PATH = os.path.join(ROOT, "settings.yml")

with codecs.open(CONFIG_PATH, encoding="utf-8") as ymlfile:
    ymlstring = ymlfile.read()

try:
    config = parse_string(ymlstring)
except PoyoException as exc:
    logging.error(exc)
else:
    logging.debug(config)


def fetch(*argv):
    return os.path.join(ROOT, *argv)


def fetch_from_home(what, *argv):
    return fetch(config[what]["dir"], *argv)


def get_from_config(*argv):
    return reduce(operator.getitem, argv, config)


def fetch_data(*argv):
    return fetch_from_home("data", get_from_config("data", *argv))


def fetch_model(name, *argv):
    if name == "current":
        name = config["model"]["current"]["name"]
    return fetch_from_home("model", name, get_from_config("model", name, *argv))


def fetch_constant(name):
    return config["constants"][name]
