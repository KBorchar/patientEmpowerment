from flask import abort
from pandas import DataFrame

def check_for_Nones(json, dicts_to_check=[], arrays_to_check=[]):
    if json is None:
        abort(415, description='Please provide a JSON file in the request body.')
    if None in json.values():
        abort(417, description='No null values in JSON fields allowed.')
    for field in dicts_to_check:
        if None in json[field].values():
            abort(417, description='No null values in JSON fields allowed.')
    for field in arrays_to_check:
        if None in json[field]:
            abort(417, description='No null values in JSON fields allowed.')

def parse_predict_request(request):
    json = request.get_json()
    check_for_Nones(json, dicts_to_check=['patient_data'], arrays_to_check=['diseases'])
    diseases = json['diseases']
    patient_data = DataFrame.from_records([json['patient_data']])
    return diseases, patient_data

def parse_get_models_request(request):
    json = request.get_json()
    check_for_Nones(json, arrays_to_check=['diseases'])
    diseases = json['diseases']
    return diseases