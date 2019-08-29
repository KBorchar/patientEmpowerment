# take a mongo collection and turn it into a dataframe.
def mongo2df(db_name, coll, limit=0):
    import pymongo
    import pandas as pd

    client = pymongo.MongoClient('localhost', 27017)
    db_name = client[db_name]
    collection = db_name[coll]
    df = pd.DataFrame(list(collection.find().limit(limit)))
    if '_id' in df.columns:
        df.drop(inplace=True, columns=["_id"])
    return df

def columns(db_name, subset_name):
    df = mongo2df(db_name, subset_name, 1)
    return df.columns.format()

def subsets(db_name):
    import pymongo

    client = pymongo.MongoClient('localhost', 27017)
    db = client[db_name]
    subset_names = db.list_collection_names()
    subsets = {}
    for s in subset_names:
        subsets[s] = subset(db_name, s)
    return subsets

def demo_subset(subset_name):
    import json

    with open('data/databases/ukbb/subsets/' + subset_name + '/demoSubset.json') as infile:
        json_string = infile.read()
        return json.loads(json_string)

def subset(db_name, subset_name):
    ensure_dir_existence(db_name, subset_name)
    return {
        "columns": columns(db_name, subset_name),
        "features_config": feature_config(db_name, subset_name),
        "models_config": models(db_name, subset_name)
    }

def models(db_name, subset_name):
    import json

    subset_path = path_name(db_name, subset_name)
    models = {}

    try:
        with open(subset_path + 'models_config.json') as infile:
            json_string = infile.read()
            models = json.loads(json_string)
    except:
        pass
    return models

def feature_config(db_name, subset_name):
    import json

    try:
        file = open(path_name(db_name, subset_name) + 'features_config.json')
        json_string = file.read()
        return json.loads(json_string)
    except:
        return {}

def database(db_name):
    return subsets(db_name)

def databases():
    import pymongo

    client = pymongo.MongoClient('localhost', 27017)
    dbs = {}
    for db in client.database_names():
        dbs[db] = database(db)
    return dbs

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

def path_name(db_name, subset_name):
    return 'data/databases/' + db_name + '/subsets/' + subset_name + '/'

def ensure_dir_existence(db_name, subset_name):
    import os

    dirname = path_name(db_name, subset_name)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    if not os.path.exists(dirname + 'objects/'):
        os.makedirs(dirname + 'objects/')

# Generate uuid in format that can be used in file-names
def short_uuid():
    import uuid
    return uuid.uuid4().hex

def dump_models_config(model_objects, columns, db_name, subset_name):
    import json

    models_config = {}
    for label, model in zip(columns, model_objects):
        feature_names = columns.copy()
        feature_names.remove(label)
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

    dirname = path_name(db_name, subset_name)
    with open(dirname + 'models_config.json', 'w') as outfile:
        json.dump(models_config, outfile)

    return models_config

# Writes models with their labels into 'data/models'
def dump_models(models, labels, db_name, subset_name):
    return dump_objects(models, labels, path_name(db_name, subset_name) + 'objects/')

# The sklearn-way to write objects to a file
# All files get placed in the 'data' directory
def dump_objects(objects, names, path=''):
    from joblib import dump
    for name, object in zip(names, objects):
        dump(object, f'{path}{name}.joblib')


# Generate a config file for the frontend features.
def dump_config(df, imputer, db_name, subset_name):
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

    with open(path_name(db_name, subset_name) + 'features_config.json', 'w') as outfile:
        json.dump(f_config, outfile)
    return f_config