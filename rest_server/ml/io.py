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

# generates dictionary of feature-coefficients and -means for a model
def get_model_dict(label):
    from ml import model_objects, imputer, dataframe_column_labels
    model = model_objects[label]
    feature_names = dataframe_column_labels.copy()
    label_index = feature_names.index(label)
    del(feature_names[label_index])
    feature_means = imputer.initial_imputer_.statistics_.copy().tolist()
    del(feature_means[label_index])
    weights = model.coef_
    model_dict = dict()
    model_dict["intercept"] = model.intercept_[0]
    features_dict = dict()
    for i, feature_name in enumerate(feature_names):
        feature_dict = dict()
        feature_dict["coef"] = weights[0][i]
        feature_dict["mean"] = feature_means[i]
        features_dict[feature_name] = feature_dict
    model_dict["features"] = features_dict
    return model_dict

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

# Writes models with their labels into 'data/models'
def dump_models(models, labels):
    return dump_objects(models, labels, 'models/')

# The sklearn-way to write objects to a file
# All files get placed in the 'data' directory
def dump_objects(objects, names, path=''):
    from joblib import dump
    for name, object in zip(names, objects):
        dump(object, f'data/{path}/{name}.joblib')

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

        # if the mean of a columns is smaller than 1, it's likely to be a binary choice between 1 and 0(e.g., asthma)...
        if imputer.initial_imputer_.statistics_[i] <= 1:
            f_config[c] = {"title": c,
                       "choices": {
                          "Yes": 1,
                          "No": 0
                        },
                       }
        # ...else it is a range choice (e.g., age), which will create a slider
        else:
            f_config[c] = {"title": c,
                       "slider_min": math.floor(minimums[i]),
                       "slider_max": math.ceil(maximums[i])
                       }

    with open('data/features.conf', 'w') as outfile:
        json.dump(f_config, outfile)
    return f_config