#! /usr/bin/env python
# author: Xinbin Huang - Vancouver School of AI
# date: Dec. 3, 2018
#
# Partially borrowed from: https://gist.github.com/W4ngatang/60c2bdb54d156a41194446737ce03e2e
# This script will download the required training data and embedding file in the project
#      - ./assets/data/train.csv
#      - ./assets/embedding/fasttext-crawl-300d-2m/crawl-300d-2M.vec
# Usage:
#    python download.py


import os
import urllib.request
import zipfile

from .utils import get_root


DIR_ROOT = get_root()
DIR_ASSETS = os.path.join(DIR_ROOT, "assets")
DATA_DIR = os.path.join(DIR_ASSETS, "data")
EMBEDDING_DIR = os.path.join(DIR_ASSETS, "embedding", "fasttext-crawl-300d-2m")


TASKS = ["Embedding"]
TASK2PATH = {
    "Embedding": "https://s3-us-west-1.amazonaws.com/fasttext-vectors/crawl-300d-2M.vec.zip"
}
TASK2DIR = {"Embedding": EMBEDDING_DIR}


def download_and_extract(task, task_url, data_dir):
    print(f"Downloading and extracting {task}")
    data_file = f"{task}.zip"
    urllib.request.urlretrieve(task_url, data_file)
    with zipfile.ZipFile(data_file) as zip_ref:
        zip_ref.extractall(data_dir)
    os.remove(data_file)
    print("\tCompleted")


def main():
    for task in TASKS:
        task_dir = TASK2DIR[task]
        task_url = TASK2PATH[task]
        if not os.path.isdir(task_dir):
            os.mkdir(task_dir)
        download_and_extract(task, task_url, task_dir)


if __name__ == "__main__":
    main()
