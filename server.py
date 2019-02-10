import argparse
import logging

from flask import Flask, request, jsonify, Response
import sys

from predict import *

logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class Server:
    def __init__(self, app: Flask, review_comparer: ReviewComparer, port: int):
        self._app = app
        self._review_comparer = review_comparer
        self._port = port
        app.route("/api/review_comparer", methods=["GET", "POST"])(self.run_comparer)

    def run_comparer(self):
        if request.method == "POST":
            data = request.get_json()
            reviews = [data["review1"], data["review2"]]

            similarity = self._review_comparer.answer_query(reviews[0], reviews[1])
            return jsonify(text=str(similarity))
        else:
            return Response(status=501)

    def serve(self):
        self._app.run(host="0.0.0.0", port=self._port, debug=True, threaded=True)


def load_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--port",
        type=int,
        default=8001,
        help="The port to listen on (default is 8001).",
    )
    return parser.parse_args()


def main():
    review_comparer = ReviewComparer()
    args = load_args()
    app = Flask(__name__)
    server = Server(app, review_comparer, args.port)
    server.serve()


if __name__ == "__main__":
    main()
