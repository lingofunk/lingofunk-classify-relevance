#! /usr/bin/env python

from scipy.spatial.distance import cosine
from train_classifier import *

DIR_ROOT = get_root()
DIR_ASSETS = os.path.join(DIR_ROOT, "assets")
DATA_DIR = os.path.join(DIR_ROOT, "yelp-data")
PATH_TO_YELP_CSV = os.path.join(DATA_DIR, "restaurant_reviews.csv")

np.random.seed(42)


class BatchGenerator:
    def __init__(self, batch_size=128, number_of_batches=None):
        self.preprocessor = Preprocess(max_features=MAX_FEATURES, maxlen=MAXLEN)
        self.restaurant_reviews = pd.read_csv(PATH_TO_YELP_CSV)
        self.restaurant_reviews = self.restaurant_reviews.groupby(["business_id"]).agg({"text": list})["text"].values
        self.n_comments_total = sum(len(restaurant) for restaurant in self.restaurant_reviews)
        self.n_restaurants = len(self.restaurant_reviews)
        self.lens_restaurants = np.array(list(map(len, self.restaurant_reviews)))
        self.fields = ["comment_0", "comment_1", "label"]
        self.batch_size = batch_size
        if number_of_batches is None:
            self.number_of_batches = np.ceil(self.n_restaurants / batch_size)

        for i in range(self.n_restaurants):
            self.preprocessor.fit_texts(self.restaurant_reviews[i])

    @staticmethod
    def compute_similarity(self, vec1, vec2):
        return cosine(vec1, vec2)

    def __generate__(self, process_target=True):
        counter = 0
        while True:
            y_batch = []
            idx_start = self.batch_size * counter
            idx_end = self.batch_size * (counter + 1)
            x_batch = []
            for i in range(idx_start, min(idx_end, self.n_restaurants)):
                n_comments = len(self.restaurant_reviews[i])
                probs = self.lens_restaurants / (self.n_comments_total - n_comments)
                probs[i] = 0

                # 1
                n_positive_examples = np.random.random_integers(2, max(3, n_comments // 10))
                positive_examples = np.random.random_integers(n_comments, size=(n_positive_examples, 2))
                for pe in positive_examples:
                    x_batch.append([self.restaurant_reviews[pe[0]][0], self.restaurant_reviews[pe[1]][0]])
                del positive_examples

                # 0
                n_negative_examples = 3 * np.random.random_integers(2, max(3, n_comments // 10))
                negative_restaurants = np.random.choice(self.n_restaurants, n_negative_examples, p=probs)
                negative_examples = []
                for restaurant in negative_restaurants:
                    negative_examples.extend(np.random.choice(self.restaurant_reviews[restaurant]))
                del negative_restaurants
                restaurant_comment_embeddings = self.preprocessor.transform_texts(self.restaurant_reviews[i])
                negative_comment_embeddings = self.preprocessor.transform_texts(negative_examples)

                negative_pairs = []

                for j in range(n_negative_examples):
                    comment_0_ind = np.random.randint(0, n_comments)
                    negative_pairs.append((self.compute_similarity(restaurant_comment_embeddings[comment_0_ind],
                                                              negative_comment_embeddings[j]),
                                           comment_0_ind, j))
                negative_pairs.sort()
                kk = n_negative_examples // 3
                for k in range(kk):
                    _, comment_0_ind, comment_1_ind = negative_pairs[k]
                    x_batch.append([self.restaurant_reviews[i][comment_0_ind], negative_examples[comment_1_ind]])
                for k in range(2 * kk, n_negative_examples):
                    _, comment_0_ind, comment_1_ind = negative_pairs[k]
                    x_batch.append([self.restaurant_reviews[i][comment_0_ind], negative_examples[comment_1_ind]])
                y_batch.append([0] * (n_negative_examples - kk))

                del negative_pairs, negative_examples, restaurant_comment_embeddings, negative_comment_embeddings
                gc.collect()
                gc.collect()
                gc.collect()
                gc.collect()

                counter += 1
                if process_target:
                    y_batch = [1] * n_positive_examples + [0] * (n_negative_examples - kk)
                    yield x_batch, y_batch
                else:
                    yield x_batch

                if process_target:
                    yield x_batch, y_batch
                else:
                    yield x_batch

                if counter == self.number_of_batches:
                    counter = 0
