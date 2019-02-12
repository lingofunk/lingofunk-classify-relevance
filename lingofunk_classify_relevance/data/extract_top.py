import json
import pandas as pd
import geojson

from lingofunk_classify_relevance.config import fetch_constant, fetch_data


def review_utility(review):
    return review["useful"] + review["funny"] + review["cool"]


def extract_geojson_and_reviews(
    city=fetch_constant("CITY"),
    top_count=fetch_constant("TOP_COUNT"),
    business_type=fetch_constant("BUSINESS_TYPE"),
    reviews_per_business=fetch_constant("REVIEWS_PER_BUSINESS"),
):

    business_ids = set()
    business = dict()
    business_review_counts = dict()
    business_reviews = dict()

    business_data = open(fetch_data("businesses"), "r", encoding="utf-8")
    reviews_data = open(fetch_data("reviews"), "r", encoding="utf-8")

    for business_json in business_data:
        info = json.JSONDecoder().decode(business_json)
        if info["city"] == city:
            business_ids.add(info["business_id"])
            business_review_counts[info["business_id"]] = info["review_count"]
            business[info["business_id"]] = {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [info["longitude"], info["latitude"]],
                },
                "properties": dict(),
            }
            for key in filter(
                lambda key: key not in ["latitude", "longitude"], info.keys()
            ):
                business[info["business_id"]]["properties"][key] = info[key]

            business[info["business_id"]]["properties"]["reviews"] = []

    business_review_counts = sorted(business_review_counts.items(), key=lambda x: -x[1])
    business_ids = set()
    for idx in range(top_count):
        business_id = business_review_counts[idx][0]
        business_ids.add(business_id)
        business_reviews[business_id] = []
    business_data.close()

    business = {business_id: business[business_id] for business_id in business_ids}

    for review_json in reviews_data:
        review = json.JSONDecoder().decode(review_json)
        business_id = review["business_id"]
        if business_id in business_ids:
            if len(business_reviews[business_id]) < reviews_per_business:
                business_reviews[business_id].append(review)
            else:
                business_reviews[business_id] = sorted(
                    business_reviews[business_id], key=review_utility, reverse=True
                )
                last_review = business_reviews[business_id][-1]
                if review_utility(review) > review_utility(last_review):
                    business_reviews[business_id][-1] = review
    reviews_data.close()

    top_reviews = []
    for business_id in business_reviews.keys():
        business[business_id]["properties"]["reviews"] = sorted(
            business_reviews[business_id], key=review_utility, reverse=True
        )
        top_reviews += business[business_id]["properties"]["reviews"]

    top_businesses = geojson.GeoJSONEncoder().encode(
        {"type": "FeatureCollection", "features": list(business.values())}
    )
    print(business.values())

    pd.DataFrame(top_reviews).to_csv(fetch_data("city"), index=False, encoding="utf-8")

    output = open(fetch_data("geojson"), "w", encoding="utf-8")
    output.write(top_businesses)
    output.close()
