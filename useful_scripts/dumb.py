from odo import odo
import pymongo
import pandas as pd

# This is a fix for fields previouslySmoking and currentlySmoking being exclusive.
# From the UKBB, if you are currently smoking, it assumes that you were not previously smoking. This script
# populates 'previouslySmoking' with 1, if 'currentlySmoking' is 1, and adjusts the -perDay numbers as well.
client = pymongo.MongoClient('localhost', 27017)
db = client.ukbb
collection = db['ahriCleaner2']

cursor = collection.find()
df = pd.DataFrame(list(cursor))
df.drop(inplace=True, columns=["_id"])

df.loc[df['currentlySmoking'] == 1, ['previouslySmoked']] = 1
df['noOfCigarettesPreviouslyPerDay'] = df.apply(
    lambda row: row['noOfCigarettesPerDay'] if row['currentlySmoking'] == 1 else row['noOfCigarettesPreviouslyPerDay'],
    axis=1
)
odo(df, db.ahriSmokingFix)
