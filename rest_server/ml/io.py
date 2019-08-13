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

def models(db_name, subset_name):
    import os
    from joblib import load

    models_path = 'data/databases/' + db_name + '/subsets/' + subset_name + '/models/'
    models = {}
    directory = os.fsencode(models_path)

    try:
        for file in os.listdir(directory):
            filename = os.fsdecode(file)
            if filename.endswith(".json"):
                models[filename.split('.')[0]] = load(models_path + filename)
    except:
        pass
    return models

def feature_config(db_name, subset_name):
    import json

    try:
        file = open('data/databases/' + db_name + '/subsets/' + subset_name + '/features.conf')
        json_string = file.read()
        return json.loads(json_string)
    except:
        return {}

def columns(db_name, subset_name):
    df = mongo2df(db_name, subset_name, 1)
    return df.columns.format()

def databases():
    import pymongo

    client = pymongo.MongoClient('localhost', 27017)
    return client.database_names()


def load_models_from_disk():
    import glob
    from joblib import load
    from ml import model_objects, model_names

    model_file_names = glob.glob("data/models/*.joblib")
    model_names = []
    for file_name in model_file_names:
        name = file_name.split('/')[-1].split('.')[0]
        model_names.append(name)

    model_objects = {}
    for name in model_names:
        model_objects[name] = load('data/models/' + name + '.joblib')

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
    parser.add_argument("-corr", "--correlator", dest="correlator",
                        help="The correlating feature to superimpose onto the visual plots")
    parser.add_argument("-db", "--database", dest="db", default='ukbb',
                        help="(mongo) database to learn from"),
    parser.add_argument("-c", "--collection", dest='collection', default='ahriCleaner2',
                        help="Collection name from the database")
    return parser.parse_args()


# Generate uuid in format that can be used in file-names
def short_uuid():
    import uuid
    return uuid.uuid4().hex

def dump_models_config(model_objects, columns, db_name, subset_name):
    import json

    models_config = {}
    for label, model in zip(columns, model_objects):
        feature_names = columns.copy()
        del (feature_names[label])
        weights = model.coef_

        model_config = {}
        model_config["intercept"] = model.intercept_[0]
        features = {}
        for i, feature_name in enumerate(feature_names):
            feature = {}
            feature["coef"] = weights[0][i]
            features[feature_name] = feature
        model_config["features"] = features
        models_config[label] = model_config

    with open('data/database/' + db_name + '/subsets/' + subset_name + '/models_config.json') as outfile:
        json.dump(models_config, outfile)

    return models_config

# Writes models with their labels into 'data/models'
def dump_models(models, labels):
    return dump_objects(models, labels, 'models/')


# The sklearn-way to write objects to a file
# All files get placed in the 'data' directory
def dump_objects(objects, names, path=''):
    from joblib import dump
    for name, object in zip(names, objects):
        dump(object, f'data/{path}/{name}.joblib')


# Generate a config file for the frontend features.
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
        mean = imputer.initial_imputer_.statistics_[i]
        if mean <= 1:
            f_config[c] = {
                "title": c,
                "choices": {
                    "Yes": 1,
                    "No": 0
                },
                "mean": mean
            }
        # ...else it is a range choice (e.g., age), which will create a slider
        else:
            f_config[c] = {
                "title": c,
                "slider_min": math.floor(minimums[i]),
                "slider_max": math.ceil(maximums[i]),
                "mean": mean
            }

    with open('data/features.conf', 'w') as outfile:
        json.dump(f_config, outfile)
    return f_config