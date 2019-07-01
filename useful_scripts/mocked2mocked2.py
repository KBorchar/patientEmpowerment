from odo import odo
import pymongo
from bson.objectid import ObjectId
import pandas as pd
import numpy as np
import pandas_profiling

client = pymongo.MongoClient('localhost', 27017)
db = client.ukbb
collection = db['ahriMocked']

cursor = collection.find()
df = pd.DataFrame(list(cursor))
df.drop(inplace=True, columns = ["_id", "neverSmoked"])

pfr = pandas_profiling.ProfileReport(df)
pfr.to_file("/tmp/df_report_mocked2.html")

wait = "once"

odo(df, db.ahriMocked2)
