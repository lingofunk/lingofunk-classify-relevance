#! /usr/bin/env python

import os
import pandas as pd
import numpy as np
import csv

from lingofunk_classify_relevance import config
from .utils import get_root

DIR_ROOT = get_root()
DIR_ASSETS = os.path.join(DIR_ROOT, "assets")
DATA_DIR = os.path.join(DIR_ASSETS, "data")

labels = [0, 1]


def split_data(df, split, dir=DATA_DIR):
    for label in labels:
        for class_name in config.classes:
            comments = df["comment_text"].where(df[class_name] == label)
            comments=comments.replace(r'^\s+$', np.nan, regex=True).dropna()
            if not comments.empty:
                filepath = os.path.join(DATA_DIR, f"{class_name}_{label}.{split}.csv")
                comments.to_csv(filepath, encoding="utf-8", index=False)


def merged_test_and_label_data(
    test_source=os.path.join(DATA_DIR, "test.csv"),
    label_source=os.path.join(DATA_DIR, "test_labels.csv"),
    dir=DATA_DIR,
):
    df_test = pd.read_csv(test_source).set_index("id")
    df_labels = pd.read_csv(label_source).set_index("id")
    return pd.concat([df_test, df_labels], axis=1, join="inner").reset_index()


def generate_data():
    df_test = merged_test_and_label_data().replace(r'^\s+$', np.nan, regex=True).dropna()
    split_data(df_test, "test")

    df_train = pd.read_csv(os.path.join(DATA_DIR, "train.csv")).replace(r'^\s+$', np.nan, regex=True).dropna()
    split_data(df_test, "train")

    df_combined = pd.concat([df_test, df_train]).replace(r'^\s+$', np.nan, regex=True).dropna()
    split_data(df_combined, "combined")

    filepath = os.path.join(DATA_DIR, "combined.csv")
    df_combined.to_csv(filepath, encoding="utf-8", index=False)


if __name__ == "__main__":
    generate_data()
