from flask import abort
from pandas import DataFrame

# takes a parsed JSON, checks the given dicts and lists for None values that came in from the outside.
# Currently we will not support Nones in the requests so we return errors.
def check_for_Nones(json, dicts_to_check=[], lists_to_check=[]):
    if json is None:
        abort(415, description='Please provide a JSON file in the request body.')
    if None in json.values():
        abort(417, description='No null values in JSON fields allowed.')
    for field in dicts_to_check:
        if None in json[field].values():
            abort(417, description='No null values in JSON fields allowed.')
    for field in lists_to_check:
        if None in json[field]:
            abort(417, description='No null values in JSON fields allowed.')

def get_db_and_collection_name(json):
    return json['db'], json['collection']

# receives labels for prediction and user_data for the instance to predict from the request JSON
def parse_predict_request(request):
    json = request.get_json()
    check_for_Nones(json, dicts_to_check=['user_data'], lists_to_check=['labels'])
    labels = json['labels']
    user_data = DataFrame.from_records([json['user_data']])
    return labels, user_data

# receives the list of labels for which the model data were requested
def parse_get_models_request(request):
    json = request.get_json()
    check_for_Nones(json, lists_to_check=['labels'])
    labels = json['labels']
    return labels

# receives the database name, collection name and list of labels for which the models are supposed to be retrained
def parse_relearn_models_request(request):
    json = request.get_json()
    check_for_Nones(json, lists_to_check=['labels'])
    db, collection = get_db_and_collection_name(json)
    labels = json['labels']
    return db, collection, labels

# receives the database and collection name for which the feature_config was requested
def parse_get_feature_config(request):
    json = request.get_json()
    check_for_Nones(json)
    db, collection = get_db_and_collection_name(json)
    return db, collection