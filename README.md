lingofunk-classify-relevance
============================

Yelp Review Relevance Classifier

Usage
-----

1. Copy YELP dataset (restaurants only) "restaurant_reviews.csv" to the "data" folder.
2. Generate pair dataset and its train/test split using yelp_dataset_generator.py.
3. Run train_classifier.py to train a model.
4. Run predict.py to predict the relevance between two reviews.

### Requirements

You will need Python 3.6 and the list of libraries:
keras, tensorflow, sklearn, numpy, pandas, scipy, pickle, csv, pathlib, logging.

Compatibility
-------------

Licence
-------

Authors
-------
