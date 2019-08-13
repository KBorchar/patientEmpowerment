from flask_app import app, request_parser

from flask import request, jsonify, abort

from ml import model_objects, dataframe_column_labels, imputer, io, learn

# predicts probabilities for given labels and user data
# requires JSON in request body, containing an array of labels and a dictionary of user_data
@app.route('/predict', methods=['POST'])
def predict():
    labels, user_data = request_parser.parse_predict_request(request)
    predictions = dict()
    for label in labels:
        data_for_label = user_data.copy()
        data_for_label.drop(columns=[label], inplace=True)
        predictions[label] = model_objects[label].predict_proba(data_for_label)[0, 1]
    response = jsonify(predictions)
    return response

# returns app config for the provided collection
# requires JSON in request body, containing a 'db' and a 'collection' field
@app.route('/feature-config', methods=['POST'])
def get_feature_config():
    db, collection = request_parser.parse_get_feature_config(request)
    df = io.mongo2df(db, collection)
    config = io.dump_config(df, imputer)
    response = jsonify(config)
    return response

# returns coefficients and means for pre-trained models on the server
# requires JSON in request body, containing an array of labels one wants the models for
@app.route('/models', methods=['POST'])
def get_models():
    labels = request_parser.parse_get_models_request(request)
    io.load_models_from_disk()
    models_dict = dict()
    for label in labels:
        models_dict[label] = io.get_model_dict(label)
    response = jsonify(models_dict)
    return response

# retrains models and returns coefficients and means of the new models
# requires JSON in request body, containing a 'db' and a 'collection' field, as well as an array of labels
# one wants new models for
@app.route('/retrain', methods=['POST'])
def retrain_models():
    db, collection, labels = request_parser.parse_retrain_models_request(request)
    try:
        df = io.mongo2df(db, collection)
    except:
        abort(503, description='Something went wrong with loading data from the mongoDB. Make sure it is running!')
    model_objects, _ = learn.train_models(df, labels)
    io.dump_models(model_objects, labels)
    imputer = learn.train_imputer(df)
    io.dump_objects([imputer], ["imputer"])
    return get_models()

@app.route('/databases', methods=['GET'])
def show_databases():
    return jsonify(io.databases())

# Redundant; same as show_subsets. Kept for robustness.
@app.route('/database/<db_name>', methods=['GET'])
def show_database(db_name):
    return jsonify(io.database(db_name))

@app.route('/database/<db_name>/subsets', methods=['GET'])
def show_subsets(db_name):
    return jsonify(io.subsets(db_name))

@app.route('/database/<db_name>/subset/<subset_name>', methods=['GET'])
def show_subset(db_name, subset_name):
    return jsonify(io.subset(db_name, subset_name))

@app.route('/database/<db_name>/subset/<subset_name>/train', methods=['POST'])
def train_subset(db_name, subset_name):
    try:
        df = io.mongo2df(db_name, subset_name)
    except:
        abort(503, description='Something went wrong with loading data from the mongoDB. Make sure it is running!')

    io.ensure_dir_existence(db_name, subset_name)
    labels = df.columns.format()
    model_objects, _ = learn.train_models(df)
    io.dump_models_config(model_objects, labels, db_name, subset_name)
    io.dump_models(model_objects, labels, db_name, subset_name)

    imputer = learn.train_imputer(df)
    io.dump_objects([imputer], ["imputer"], io.path_name(db_name, subset_name) + 'objects/')

    io.dump_config(df, imputer, db_name, subset_name)

    return show_subset(db_name, subset_name)