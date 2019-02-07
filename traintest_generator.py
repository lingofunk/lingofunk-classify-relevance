#! /usr/bin/env python

from train_classifier_attn import *

DIR_ROOT = get_root()
DIR_ASSETS = os.path.join(DIR_ROOT, "assets")
DATA_DIR = os.path.join(DIR_ROOT, "yelp-data")
PATH_TO_YELP_CSV = os.path.join(DATA_DIR, "restaurant_reviews.csv")

np.random.seed(42)


def split_data(test_size=0.25):
    data = pd.read_csv(PATH_TO_YELP_CSV)
    data_train, data_test = train_test_split(data, random_state=42, test_size=test_size)
    del data
    data_train.to_csv(os.path.join(DATA_DIR, "reviews_train.csv"))
    data_test.to_csv(os.path.join(DATA_DIR, "reviews_test.csv"))
    del data_train, data_test


if __name__ == "__main__":
    split_data()
