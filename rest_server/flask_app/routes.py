from flask_app import app, request_parser

from flask import request, jsonify

from ml import models, model_objects, dataframe_column_labels, imputer, helpers, learn


# requires JSON in request body, containing target disease and patient
# {
#   diseases: ['...', '...']
#   patient_data: {...}
# }
@app.route('/predict', methods=['GET'])
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
@app.route('/config', methods=['GET'])
def get_config():
    db, collection = request_parser.parse_get_config(request)
    df = helpers.mongo2df(db, collection)
    config = helpers.dump_config(df, imputer)
    response = jsonify(config)
    return response

# returns models currently in use
@app.route('/models', methods=['GET'])
def get_models():
    labels = request_parser.parse_get_models_request(request)
    models_dict = dict()
    for label in labels:
        models_dict[label] = models.get_model_dict(label)
    response = jsonify(models_dict)
    return response

# retrains models and returns new models.
@app.route('/retrain', methods=['GET'])
def retrain_models():
    db, collection, labels = request_parser.parse_relearn_models_request(request)
    df = helpers.mongo2df(db, collection) # TODO: try catch block, return error code if this fails i guess
    model_objects, _ = learn.train_models(df, labels, None)
    helpers.dump(model_objects, labels)
    imputer = learn.train_imputer(df)
    helpers.dump([imputer], ["imputer"])
    return get_models()