from odo import odo
import pymongo
from bson.objectid import ObjectId
import pandas as pd
import numpy as np
import pandas_profiling

client = pymongo.MongoClient('localhost', 27017)
db = client.ukbb
collection = db['ahriCleaner']

cursor = collection.find()
df = pd.DataFrame(list(cursor))
df.drop(inplace=True, columns = ["_id"])

#choose all COPDs, diabetes', TB's
#slap 'em on there

df2 = pd.DataFrame()
diagnoses = ["COPD", "diabetes", "tuberculosis"]

temp_df = pd.DataFrame()
temp_df = df.loc[(df[diagnoses[0]] == 1) | (df[diagnoses[1]] == 1) | (df[diagnoses[2]] == 1)]
df2 = df2.append(temp_df)

#fill up to 20k (+14k)

df_healthy = pd.DataFrame()
df_healthy = df.loc[(df[f'{diagnoses[0]}'] != 1) & (df[f'{diagnoses[1]}'] != 1) & (df[f'{diagnoses[2]}'] != 1)]
df_healthy = df_healthy.sample(n=14156)

df2 = df2.append(df_healthy)

pfr = pandas_profiling.ProfileReport(df2)
pfr.to_file("/tmp/df_report_mocked2.html")

wait = "once"

odo(df2, db.ahriMocked)
