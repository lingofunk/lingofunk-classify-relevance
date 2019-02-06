#! /usr/bin/env python

from scipy.spatial.distance import cosine
import csv
from multiprocessing.pool import Pool

from train_classifier import *

DIR_ROOT = get_root()
DIR_ASSETS = os.path.join(DIR_ROOT, "assets")
DATA_DIR = os.path.join(DIR_ROOT, "yelp-data")
PATH_TO_YELP_CSV = os.path.join(DATA_DIR, "restaurant_reviews.csv")

np.random.seed(42)


def split_data(test_size=0.25):
    data = pd.read_csv(os.path.join(DATA_DIR, "restaurant_reviews_pairs.csv"))
    data_train, data_test = train_test_split(data, random_state=42, test_size=test_size)
    del data
    data_train.to_csv(os.path.join(DATA_DIR, "rel-train.csv"))
    data_test.to_csv(os.path.join(DATA_DIR, "rel-test.csv"))
    del data_train, data_test


import gc, sys


def generate_data():
    def compute_similarity(vector_1, vector_2):
        return cosine(vector_1, vector_2)

    preprocessor = Preprocess(max_features=MAX_FEATURES, maxlen=MAXLEN)
    restaurant_reviews = pd.read_csv(PATH_TO_YELP_CSV)
    restaurant_reviews = restaurant_reviews.groupby(["business_id"]).agg({"text": list})["text"].values

    n_comments_total = sum(len(restaurant) for restaurant in restaurant_reviews)
    n_restaurants = len(restaurant_reviews)

    preprocessor.fit_texts([restaurant_reviews[i] for i in range(n_restaurants)])

    print("n_restaurants: ", n_restaurants)
    print("n_comments_total", n_comments_total)

    lens_restaurants = np.array(list(map(len, restaurant_reviews)))

    fields = ["comment_0", "comment_1", "label"]

    for size, o in sorted([(sys.getsizeof(o), o) for o in gc.get_objects()], key=lambda p: p[0], reverse=True)[:1]:
        print(size, str(o)[:50])
        print()

    with open(os.path.join(DATA_DIR, "restaurant_reviews_pairs.csv"), 'w') as f:
        writer = csv.writer(f)
        writer.writerow(fields)

    for i in range(n_restaurants):
        with open(os.path.join(DATA_DIR, "restaurant_reviews_pairs.csv"), 'a') as f:
            writer = csv.writer(f)
            n_comments = len(restaurant_reviews[i])
            if i % 100 == 0:
                print(f' restaurant # {i}')
                for size, o in sorted([(sys.getsizeof(o), o) for o in gc.get_objects()], key=lambda p: p[0],
                                      reverse=True)[:5]:
                    print(size, str(o)[:50])
                    print()
            probs = lens_restaurants / (n_comments_total - n_comments)
            probs[i] = 0
            # 1
            n_positive_examples = np.random.random_integers(2, max(3, n_comments // 10))
            positive_examples = np.random.random_integers(n_comments, size=(n_positive_examples, 2))
            for pe in positive_examples:
                comment_0 = restaurant_reviews[pe[0]][0]
                comment_1 = restaurant_reviews[pe[1]][0]
                writer.writerow([comment_0, comment_1, 1])
            del positive_examples

            # 0
            n_negative_examples = 3 * np.random.random_integers(2, max(3, n_comments // 10))
            negative_restaurants = np.random.choice(n_restaurants, n_negative_examples, p=probs)
            negative_examples = [np.random.choice(restaurant_reviews[restaurant]) for restaurant in negative_restaurants]
            del negative_restaurants
            restaurant_comment_embeddings = preprocessor.transform_texts(restaurant_reviews[i])
            negative_comment_embeddings = preprocessor.transform_texts(negative_examples)

            negative_pairs = []

            for j in range(n_negative_examples):
                comment_0_ind = np.random.randint(0, n_comments)
                negative_pairs.append((compute_similarity(restaurant_comment_embeddings[comment_0_ind],
                                                          negative_comment_embeddings[j]),
                                       comment_0_ind, j))
            negative_pairs.sort()
            for k in range(n_negative_examples // 3):
                _, comment_0_ind, comment_1_ind = negative_pairs[k]
                writer.writerow([restaurant_reviews[i][comment_0_ind], negative_examples[comment_1_ind], 0])
            for k in range(2 * n_negative_examples // 3, n_negative_examples):
                _, comment_0_ind, comment_1_ind = negative_pairs[k]
                writer.writerow([restaurant_reviews[i][comment_0_ind], negative_examples[comment_1_ind], 0])
            del negative_pairs, negative_examples, restaurant_comment_embeddings, negative_comment_embeddings


if __name__ == "__main__":
    generate_data()
