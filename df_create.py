# pymongo
# http://api.mongodb.com/python/current/
# http://api.mongodb.com/python/current/tutorial.html

### python:
import pymongo
from bson.objectid import ObjectId
import pprint
import pdb
import pandas as pd
import numpy as np

client = pymongo.MongoClient('localhost', 27017) # connects to mongodb server
db = client.ukbb # select database on server ('use ukbb' in shell)
collection = db['ahriWithGeoArray']
pprint.pprint(collection.find_one()) # returns one document from temp4 collection ('db.temp4.findOne()' in shell
pprint.pprint(collection.find_one({ # mongo documents are noted as python dictionaries, arrays as python arrays. 
		'yearOfBirth': 1950,
		'monthOfBirth': 3,
		'dateOfAttendingAssessmentCentre': ['2008-09-02', 'NA', 'NA']
	}))

cursor = collection.find().limit(1000) # TODO: dont limit in production
df = pd.DataFrame(list(cursor))

def flatten(np_series):
    flat_list = np.Series()
    for list in np_series:
        for element in list:
            flat_list.append(element)
    return flat_list

def analyze_NAs(df):
    column_count = df.shape[0]
    NA_percentages = np.zeros(column_count)
    for column_index in range(column_count):
        column = df.iloc[:, column_index]
        pdb.set_trace()
        counts = column.value_counts(normalize=True, dropna=False)
        pdb.set_trace()
        #if counts[0] is list:

        if "NA" in counts:
            NA_percentages[column_index] = counts["NA"]

    return

analyze_NAs(df)
