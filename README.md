lingofunk-classify-relevance
============================

Yelp Review Relevance Classifier

Usage
-----

1. Copy YELP dataset (restaurants only) "restaurant_reviews.csv" to the "data" folder.
2. Generate pair dataset using yelp_dataset_generator.py.
3. Generate train/test split using traintest_generator.py.
4. Run train_classifier.py or train_classifier_attn.py to train a model.
5. Run predict.py (onlt for a model without attn.---bug somewhere) to predict the relevance between two reviews.

### Requirements

You will need Python 3.6 and the list of libraries:
keras, tensorflow, sklearn, numpy, pandas, scipy, pickle, csv, pathlib, logging, bpemb, gensim.

Compatibility
-------------

Licence
-------

Authors
-------
