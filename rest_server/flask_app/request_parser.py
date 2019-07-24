from flask import abort
import flask.abort
from pandas import DataFrame

def check_for_Nones(json, dicts_to_check=[], arrays_to_check=[]):
    if json is None:
        flask.abort(415, description='Please provide a JSON file in the request body.')
    if None in json.values():
        flask.abort(417, description='No null values in JSON fields allowed.')
    for field in dicts_to_check:
        if None in json[field].values():
            flask.abort(417, description='No null values in JSON fields allowed.')
    for field in arrays_to_check:
        if None in json[field]:
            flask.abort(417, description='No null values in JSON fields allowed.')

def get_db_and_collection_name(json):
    return json['db'], json['collection']

def parse_predict_request(request):
    json = request.get_json()
    check_for_Nones(json, dicts_to_check=['user_data'], arrays_to_check=['labels'])
    labels = json['labels']
    user_data = DataFrame.from_records([json['user_data']])
    return labels, user_data

def parse_get_models_request(request):
    json = request.get_json()
    check_for_Nones(json, arrays_to_check=['labels'])
    labels = json['labels']
    return labels

def parse_relearn_models_request(request):
    json = request.get_json()
    check_for_Nones(json, arrays_to_check=['labels'])
    db, collection = get_db_and_collection_name(json)
    labels = json['labels']
    return db, collection, labels

def parse_get_feature_config(request):
    json = request.get_json()
    check_for_Nones(json)
    db, collection = get_db_and_collection_name(json)
    return db, collection