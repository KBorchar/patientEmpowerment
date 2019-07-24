# initializes ml package
from joblib import load
import json
import glob

# on import of this package, load all models stored on the server's disk
# prevents having to retrain all models every time we restart the server
model_file_names = glob.glob("data/models/*.joblib")
model_names = []
for file_name in model_file_names:
    name = file_name.split('/')[-1].split('.')[0]
    model_names.append(name)

model_objects = {}
for name in model_names:
    model_objects[name] = load('data/models/' + name + '.joblib')

imputer = load('data/imputer.joblib')

with open('data/columns.txt') as json_file:
    dataframe_column_labels = json.load(json_file)