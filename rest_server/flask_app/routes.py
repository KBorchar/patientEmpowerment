from flask_app import app, request_parser

from flask import request, jsonify

from ml import models, model_objects, dataframe_column_labels, imputer, helpers, learn

@app.route('/')
def index():
    return "Hello, World!"

# requires JSON in request body, containing target disease and patient
# {
#   diseases: ['...', '...']
#   patient_data: {...}
# }
@app.route('/predict', methods=['GET'])
def predict():
    diseases, patient_data = request_parser.parse_predict_request(request)
    predictions = dict()
    for disease in diseases:
        data_for_disease = patient_data.copy()
        data_for_disease.drop(columns=[disease], inplace=True)
        predictions[disease] = model_objects[disease].predict(data_for_disease)[0]
    response = jsonify(predictions)
    return response

# returns models currently in use
@app.route('/models', methods=['GET'])
def get_models():
    diseases = request_parser.parse_get_models_request(request)
    models_dict = dict()
    for disease in diseases:
        models_dict[disease] = models.get_model_dict(disease)
    response = jsonify(models_dict)
    return response

# retrains models and returns new models.
@app.route('/retrain', methods=['GET'])
def retrain_models():
    diseases = request_parser.parse_relearn_models_request(request)
    df = helpers.mongo2df('ukbb', 'ahriMocked') # TODO: try block, return error code if this fails i guess
    model_objects, _ = learn.train_models(df, diseases, None)
    helpers.dump(model_objects, diseases)
    imputer = learn.train_imputer(df)
    helpers.dump([imputer], ["imputer"])
    return get_models()