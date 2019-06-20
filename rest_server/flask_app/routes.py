from flask_app import app, request_parser

from flask import request, jsonify

from ml import models, model_objects, dataframe_column_labels, imputer

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
        predictions[disease] = model_objects[disease].predict(patient_data)[0]
        print(disease, predictions[disease])
    response = jsonify(predictions)
    return response

@app.route('/models', methods=['GET'])
def get_models():
    diseases = request_parser.parse_get_models_request(request)
    models_dict = dict()
    for disease in diseases:
        models_dict[disease] = models.get_model_dict(disease)
    response = jsonify(models_dict)
    return response