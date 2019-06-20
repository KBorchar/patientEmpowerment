from flask_app import app, request_parser

from flask import request, jsonify

from ml import models, dataframe_column_labels, imputer, data_cleaner

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
        predictions[disease] = models[disease].predict(patient_data)[0]
        print(disease, predictions[disease])
    response = jsonify(predictions)
    return response

@app.route('/models', methods=['GET'])
def get_models():
    diseases = request_parser.parse_get_models_request(request)
    models_response = dict()
    for disease in diseases:
        model = models[disease]
        features = dataframe_column_labels.copy()
        label_index = features.index(disease)
        del(features[label_index])
        means = imputer.initial_imputer_.statistics_.copy().tolist()
        del(means[label_index])
        weights = model.coef_
        features_dict = dict()
        for i, feature in enumerate(features):
            feature_dict = dict()
            feature_dict["coef"] = weights[0][i]
            feature_dict["mean"] = means[i]
            features_dict[feature] = feature_dict
        models_response[disease] = features_dict
    return jsonify(models_response)