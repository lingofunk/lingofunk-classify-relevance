import pandas as pd
from sklearn.model_selection import train_test_split
from lingofunk_classify_relevance.config import fetch_data


def split_data(test_size=0.25):
    data = pd.read_csv(fetch_data("source"))
    data_train, data_test = train_test_split(data, random_state=42, test_size=test_size)
    del data
    data_train.to_csv(fetch_data("train"))
    data_test.to_csv(fetch_data("test"))
    del data_train, data_test


if __name__ == "__main__":
    split_data()
