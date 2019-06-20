# initializes ml module
# if we ever want to predict serverside, this module is the place for that
from joblib import load
import json

model_names = ["COPD", "asthma", "diabetes", "tuberculosis"]
models = {}
for name in model_names:
    models[name] = load('data/models/' + name + '.joblib')

imputer = load('data/models/imputer.joblib')

dataframe_column_labels = None
with open('data/columns.txt') as json_file:
    dataframe_column_labels = json.load(json_file)