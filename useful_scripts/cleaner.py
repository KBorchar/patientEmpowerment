from odo import odo
import pymongo
import pandas as pd
import numpy as np

# Remove empty fields, change fields with weird encodings into own columns with binary input.
# E.g. SmokingStatus had encoding with arbitrary meanings (0 => never, 1 => in the past, etc.).
# In order to use them later, we needed own columns for them.
#
# The offset is only there because our VM could get around 100000 entries in one iteration.
#
# Also remove any occurrences of "patient wouldn't say" (== -3) or "didnt know" (== -1)

offset = int(input("Enter offset to start on:"))
client = pymongo.MongoClient('localhost', 27017)  # connects to mongodb server
db = client.ukbb  # select database on server (equivalent: 'use ukbb' in mongo shell)
collection = db['ahriML']

cursor = collection.find().skip(offset).limit(100000)  #this limit is close to what our VM could manage ram-wise.
df = pd.DataFrame(list(cursor))
df.replace("NA", np.nan, inplace=True) #In the UKBB, NaNs are filled with Strings "NA".

# This actually drops 99% of all NaNs: Lines with COPD filled out were generally filled in all other columns as well.
df = df.dropna(subset=["COPD"])

df.drop(inplace=True, columns=["_id", "patientId"]) #Let's  not learn on IDs

# The new columns. They will contain booleans.
df["neverSmoked"] = np.nan
df["previouslySmoked"] = np.nan
df["currentlySmoking"] = np.nan

# Seems these would only be filled when the patient had been smoking.
# We assume no value means patient is and/or was non-smoker.
df.loc[np.isnan(df['noOfCigarettesPreviouslyPerDay']), ['noOfCigarettesPreviouslyPerDay']] = 0
df.loc[np.isnan(df['noOfCigarettesPerDay']), ['noOfCigarettesPerDay']] = 0

# less than one per day, don't know or won't tell; barely any occurrences.
df.loc[df['noOfCigarettesPerDay'] < 0, ['noOfCigarettesPerDay']] = np.nan

df.loc[df['smokingStatus'] == -3] = np.nan
df.loc[df['smoking'] == -3] = np.nan

# Kind of like "IF smokingStatus == 0 THEN neverSmoked = 1, others = 0."
df.loc[df['smokingStatus'] == 0, ['neverSmoked']] = 1
df.loc[df['smokingStatus'] == 0, ['previouslySmoked']] = 0
df.loc[df['smokingStatus'] == 0, ['currentlySmoking']] = 0

df.loc[df['smokingStatus'] == 1, ['neverSmoked']] = 0
df.loc[df['smokingStatus'] == 1, ['previouslySmoked']] = 1
df.loc[df['smokingStatus'] == 1, ['currentlySmoking']] = 0

df.loc[df['smokingStatus'] == 2, ['neverSmoked']] = 0
df.loc[df['smokingStatus'] == 2, ['previouslySmoked']] = 0
df.loc[df['smokingStatus'] == 2, ['currentlySmoking']] = 1

# Smoking == 2 is 'only occasionally', which we basically decided means one per day.
df.loc[df['smoking'] == 2, ['noOfCigarettesPerDay']] = 1
df.loc[df['smoking'] == 2, ['neverSmoked']] = 0
df.loc[df['smoking'] == 2, ['previouslySmoked']] = 0
df.loc[df['smoking'] == 2, ['currentlySmoking']] = 1

df.loc[df['alcoholFrequency'] == -3, ['alcoholFrequency']] = np.nan

df.loc[df['diabetes'] == -1, ['diabetes']] = 0
df.loc[df['diabetes'] == -3, ['diabetes']] = np.nan

df.loc[df['wheezeInChestInLastYear'] == -1, ['wheezeInChestInLastYear']] = 0
df.loc[df['wheezeInChestInLastYear'] == -3, ['wheezeInChestInLastYear']] = np.nan

df['systolicBloodPressure'] = df[['systolicBloodPressure0', 'systolicBloodPressure1']].mean(axis=1)
df['diastolicBloodPressure'] = df[['diastolicBloodPressure0', 'diastolicBloodPressure1']].mean(axis=1)

df = df.dropna(subset=['diastolicBloodPressure', 'weight', 'height', 'noOfCigarettesPerDay', 'alcoholFrequency', 'diabetes', 'wheezeInChestInLastYear', 'smoking', 'smokingStatus'])
df.drop(inplace=True, columns=['systolicBloodPressure0', 'systolicBloodPressure1', 'diastolicBloodPressure0', 'diastolicBloodPressure1', 'smoking', 'smokingStatus', 'stoppedSmokingFor6Months'])

# Add df to ahriCleaner in Mongo.
odo(df, db.ahriCleaner)
