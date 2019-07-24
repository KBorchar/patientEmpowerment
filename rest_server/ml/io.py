# take a mongo collection and turn it into a dataframe.
def mongo2df(db, coll, limit=0):
    import pymongo
    import pandas as pd

    client = pymongo.MongoClient('localhost', 27017)
    db = client[db]
    collection = db[coll]
    df = pd.DataFrame(list(collection.find().limit(limit)))
    if '_id' in df.columns:
        df.drop(inplace=True, columns=["_id"])
    return df

# argument parsing for the command line
def get_args():
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("-o", "--output", action='store_true', help="Draw a confusion matrix for various scores.")
    parser.add_argument("-l", "--labels", dest="labels", default='COPD asthma diabetes tuberculosis',
                        help="Labels to predict on, separated by spaces, e.g.: \"COPD asthma diabetes\".")
    parser.add_argument("-c", "--correlator", dest="correlator",
                        help="The correlating feature that you want to superimpose onto the visual plots.")
    parser.add_argument("-db", "--database", dest="db", default='ukbb',
                        help="mongo collection to learn from"),
    parser.add_argument("-coll", "--collection", dest='collection', default='ahriCleaner2',
                        help="collection name from the database")
    return parser.parse_args()

# Generate uuid in format that can be used in file-names
def short_uuid():
    import uuid
    return uuid.uuid4().hex

# The sklearn-way to write models to a file
def dump_models(models, labels):
    from joblib import dump
    for i, t in enumerate(models):
        dump(t, f'data/models/{labels[i]}.joblib')

# Generate a config file for the frontend.
def dump_config(df, imputer):
    import json
    import math

    columns = df.columns.format()

    # Min and max will inform number ranges for sliders in the frontend
    minimums = df.min()
    maximums = df.max()
    f_config = {}
    for i, c in enumerate(columns):

        # if the mean of a columns is smaller than 1, it's likely to be a binary choice between 1 and 0.
        if imputer.initial_imputer_.statistics_[i] <= 1:
            f_config[c] = {"title": c,
                       "choices": {
                          "Yes": 1,
                          "No": 0
                        },
                       }
        # else it is a range choice, which will create a slider
        else:
            f_config[c] = {"title": c,
                       "slider_min": math.floor(minimums[i]),
                       "slider_max": math.ceil(maximums[i])
                       }

    with open('data/features.conf', 'w') as outfile:
        json.dump(f_config, outfile)
    return f_config






