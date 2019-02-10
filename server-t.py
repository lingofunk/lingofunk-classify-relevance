import argparse
import logging

from flask import Flask, request, jsonify, Response
import sys

from TownTextExtractor import *

logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class Server:
    def __init__(self, app: Flask, tte: TownTextExtractor, port: int):
        self._app = app
        self._tte = tte
        self._port = port
        app.route("/api/get_similar", methods=["GET", "POST"])(self.run)

    def run(self):
        if request.method == "POST":
            data = request.get_json()
            restaurants = self._tte.get_heatmap_for_restaurant(data["id"])
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
        default=8001,
        help="The port to listen on (default is 8001).",
    )
    return parser.parse_args()


def main():
    tte = TownTextExtractor()
    tte.load_similarity_matrix()
    args = load_args()
    app = Flask(__name__)
    server = Server(app, tte, args.port)
    server.serve()


if __name__ == "__main__":
    main()
