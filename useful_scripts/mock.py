from odo import odo
import pymongo
import pandas as pd
import pandas_profiling

# UKBB was generally pretty healthy. To make the dataset more similar to AHRI's, we choose people with
# TB, diabetes, COPD in amounts that mirror the official percentages of South Africa (NOT KwaZulu-Natal only).
# In addition, we limit the amount of entries to 20000, which mirrors the amount Vukuzazi has after 2 years.
client = pymongo.MongoClient('localhost', 27017)
db = client.ukbb
collection = db['ahriCleaner']

cursor = collection.find()
df = pd.DataFrame(list(cursor))
df.drop(inplace=True, columns=["_id"])

df2 = pd.DataFrame()
diagnoses = ["COPD", "diabetes", "tuberculosis"]

# Find entries with diagnoses.
temp_df = pd.DataFrame()
temp_df = df.loc[(df[diagnoses[0]] == 1) | (df[diagnoses[1]] == 1) | (df[diagnoses[2]] == 1)]
df2 = df2.append(temp_df)

# Fill up to 20k entries
df_healthy = pd.DataFrame()
df_healthy = df.loc[(df[f'{diagnoses[0]}'] != 1) & (df[f'{diagnoses[1]}'] != 1) & (df[f'{diagnoses[2]}'] != 1)]
df_healthy = df_healthy.sample(n=14156)

df2 = df2.append(df_healthy)

pfr = pandas_profiling.ProfileReport(df2)
pfr.to_file("/tmp/df_report_mocked2.html")

# Add df2 to ahriMocked in the Mongo.
odo(df2, db.ahriMocked)
