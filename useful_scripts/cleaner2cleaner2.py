from odo import odo
import pymongo
import pandas as pd
import pandas_profiling

# Removes the neverSmoked field, as it is redundant when "currentlySmoking" and "PreviouslySmoking" exist.
client = pymongo.MongoClient('localhost', 27017)
db = client.ukbb
collection = db['ahriCleaner']

cursor = collection.find()
df = pd.DataFrame(list(cursor))
df.drop(inplace=True, columns=["_id", "neverSmoked"])

pfr = pandas_profiling.ProfileReport(df)
pfr.to_file("/tmp/df_report_cleaner2.html")

odo(df, db.ahriCleaner2)
