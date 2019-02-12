import argparse
import logging

from flask import Flask, request, jsonify, Response
import sys

from lingofunk_classify_relevance.data.city_analyst import CityAnalyst

logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class Server:
    def __init__(self, app: Flask, city_analyst: CityAnalyst, port: int):
        self._app = app
        self._city_analyst = city_analyst
        self._port = port
        app.route("/api/review_comparer", methods=["GET", "POST"])(self.run_comparer)
        app.route("/api/get_similar", methods=["GET", "POST"])(self.run_similar)
        app.route("/api/get_unique", methods=["GET", "POST"])(self.run_unique)

    def run_comparer(self):
        if request.method == "POST":
            data = request.get_json()
            reviews = [data["review1"], data["review2"]]
            similarity = self._city_analyst.comparer.answer_query(
                reviews[0], reviews[1]
            )
            return jsonify(text=str(similarity))
        else:
            return Response(status=501)

    def run_similar(self):
        if request.method == "POST":
            data = request.get_json()
            restaurants = self._city_analyst.get_heatmap_for_restaurant(data["id"])
            return jsonify(restaurants=restaurants)
        else:
            return Response(status=501)

    def run_unique(self):
        if request.method == "GET":
            restaurants = self._city_analyst.get_unique_restaurants()
            return jsonify(restaurants=restaurants)
        else:
            return Response(status=501)

    def serve(self):
        self._app.run(host="0.0.0.0", port=self._port, debug=True, threaded=True)


def load_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="The port to listen on (default is 8001).",
    )
    return parser.parse_args()


def main():
    tte = CityAnalyst()
    tte.load_similarity_matrix()
    args = load_args()
    app = Flask(__name__)
    server = Server(app, tte, args.port)
    server.serve()


if __name__ == "__main__":
    main()
