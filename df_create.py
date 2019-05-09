# pymongo
# http://api.mongodb.com/python/current/
# http://api.mongodb.com/python/current/tutorial.html

### python:
import pymongo
from bson.objectid import ObjectId
import pprint
import pdb
import pandas as pd

client = pymongo.MongoClient('localhost', 27017) # connects to mongodb server
db = client.ukbb # select database on server ('use ukbb' in shell)
collection = db['ahriWithGeoArray']
pprint.pprint(collection.find_one()) # returns one document from temp4 collection ('db.temp4.findOne()' in shell
pprint.pprint(collection.find_one({ # mongo documents are noted as python dictionaries, arrays as python arrays. 
		'yearOfBirth': 1950,
		'monthOfBirth': 3,
		'dateOfAttendingAssessmentCentre': ['2008-09-02', 'NA', 'NA']
	}))

cursor = collection.find().limit(1000)
df = pd.DataFrame(list(cursor))

pdb.set_trace()
