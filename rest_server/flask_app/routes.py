from flask_app import app, request_parser

from flask import request, jsonify
import flask.abort

from ml import models, model_objects, dataframe_column_labels, imputer, io, learn

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
    models_dict = dict()
    for label in labels:
        models_dict[label] = models.get_model_dict(label)
    response = jsonify(models_dict)
    return response

# retrains models and returns coefficients and means of the new models
# requires JSON in request body, containing a 'db' and a 'collection' field, as well as an array of labels
# one wants new models for
@app.route('/retrain', methods=['POST'])
def retrain_models():
    db, collection, labels = request_parser.parse_relearn_models_request(request)
    try:
        df = io.mongo2df(db, collection)
    except:
        flask.abort(503, description='Something went wrong with loading data from the mongoDB. Make sure it is running!')
    model_objects, _ = learn.train_models(df, labels, None)
    io.dump_models(model_objects, labels)
    imputer = learn.train_imputer(df)
    io.dump_objects([imputer], ["imputer"])
    return get_models()