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
    parser.add_argument("-coll", "--collection", dest='collection', default='ahriMocked2',
                        help="collection name from the database")
    return parser.parse_args()


def short_uuid():
    import uuid
    return uuid.uuid4().hex


def dump(things, names):
    from joblib import dump
    for i, t in enumerate(things):
        dump(t, f'data/models/{names[i]}.joblib')


def dump_JSON(columns):
    import json
    with open('data/columns.txt', 'w') as outfile:
        json.dump(columns, outfile)


def dump_config(df, imputer):
    import json
    import math

    columns = df.columns.format()
    minimums = df.min()
    maximums = df.max()
    f_config = {}
    for i, c in enumerate(columns):
        if imputer.initial_imputer_.statistics_[i] <= 1:
            f_config[c] = {"title": c,
                       "choices": {
                          "Yes": 1,
                          "No": 0
                        },
                       }
        else:
            f_config[c] = {"title": c,
                       "slider_min": math.floor(minimums[i]),
                       "slider_max": math.ceil(maximums[i])
                       }

    with open('data/features.conf', 'w') as outfile:
        json.dump(f_config, outfile)
    return f_config






