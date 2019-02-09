from predict import *

DATASET_CSV = os.path.join(DATA_DIR, "Brooklyn_dataset.csv")


class TownTextExtractor:
    def __init__(self):
        self.restaurant_reviews = pd.read_csv(DATASET_CSV, sep="\\")

        self.restaurant_reviews = self.restaurant_reviews.groupby(["business_id"]).agg({"text": list})

        self.id2rest = self.restaurant_reviews["business_id"].values
        self.rest2id = dict()
        for i, rest in enumerate(self.id2rest):
            self.rest2id[rest] = i

        print(self.rest2id)
        print(self.id2rest)

        self.restaurant_reviews = self.restaurant_reviews["text"].values
        self.n_comments_total = sum(len(restaurant) for restaurant in self.restaurant_reviews)
        self.n_restaurants = len(self.restaurant_reviews)
        self.lens_restaurants = np.array(list(map(len, self.restaurant_reviews)))

        self.comparer = ReviewComparer()

        self.similarity_matrix = np.zeros(shape=(self.n_restaurants, self.n_restaurants))
        self.uniqueness = np.zeros(shape=(self.n_restaurants,))
        self.uniqueness_args = np.zeros(shape=(self.n_restaurants,))
        self.uniqueness_sorted = np.zeros(shape=(self.n_restaurants,))

    def compute_similarity_matrix(self):
        for i in range(self.n_restaurants):
            restaurants_i = self.restaurant_reviews[i]
            for j in range(self.n_restaurants):
                n_ij = self.lens_restaurants[i] * self.lens_restaurants[j]
                sim_ij = 0
                restaurants_j = self.restaurant_reviews[j]
                for r_i in restaurants_i:
                    for r_j in restaurants_j:
                        sim_ij += self.comparer.answer_query(r_i, r_j)
                sim_ij /= n_ij
                self.similarity_matrix[i][j] = sim_ij
        self.uniqueness = np.sum(self.similarity_matrix)
        self.uniqueness_args = np.argsort(self.uniqueness)
        self.uniqueness_sorted = self.uniqueness[self.uniqueness_args]

    def get_heatmap_for_restaurant_id(self, i):
        return self.similarity_matrix[i] / sum(self.similarity_matrix[i])

    def get_heatmap_for_restaurant_name(self, rest):
        return self.get_heatmap_for_restaurant_id(self.rest2id[rest])

    def get_heatmap_for_restaurant(self, q):
        if isinstance(g, int):
            return zip(self.id2rest, self.get_heatmap_for_restaurant_id(q))
        elif isinstance(g, str):
            return zip(self.id2rest, self.get_heatmap_for_restaurant_name(q))
        else:
            print("Unexpected type of input.")

    def get_unique_restaurants(self):
        return zip(self.uniqueness_args, self.uniqueness_sorted)
