# pymongo
# http://api.mongodb.com/python/current/
# http://api.mongodb.com/python/current/tutorial.html

### python:
import pymongo
from bson.objectid import ObjectId
import pandas as pd
import numpy as np
import pandas_profiling
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import svm
from sklearn.ensemble import GradientBoostingRegressor


client = pymongo.MongoClient('localhost', 27017)  # connects to mongodb server
db = client.ukbb  # select database on server ('use ukbb' in shell)
collection = db['ahriML']
#pprint.pprint(collection.find_one())  # returns one document from temp4 collection ('db.temp4.findOne()' in shell
#pprint.pprint(collection.find_one({  # mongo documents are noted as python dictionaries, arrays as python arrays.
#    'yearOfBirth': 1950,
#    'monthOfBirth': 3,
#    'dateOfAttendingAssessmentCentre': ['2008-09-02', 'NA', 'NA']
#}))

cursor = collection.find().limit(100000)  # TODO: dont limit once we have more RAM
df = pd.DataFrame(list(cursor))
df.replace("NA", np.nan, inplace=True)
df = df.dropna(subset=["COPD"])
df.drop(inplace=True, columns = ["_id", "patientId"])

df["neverSmoked"] = np.nan
df["previouslySmoked"] = np.nan
df["currentlySmoking"] = np.nan

df.loc[np.isnan(df['noOfCigarettesPreviouslyPerDay']), ['noOfCigarettesPreviouslyPerDay']] = 0
df.loc[np.isnan(df['noOfCigarettesPerDay']), ['noOfCigarettesPerDay']] = 0 # assuming no value means patient is non-smoker
df.loc[df['noOfCigarettesPerDay'] < 0, ['noOfCigarettesPerDay']] = np.nan # less than one per day, don't know or won't tell; barely any occurences

df.loc[df['smokingStatus'] == -3] = np.nan
df.loc[df['smoking'] == -3] = np.nan

df.loc[df['smokingStatus'] == 0, ['neverSmoked']] = 1
df.loc[df['smokingStatus'] == 0, ['previouslySmoked']] = 0
df.loc[df['smokingStatus'] == 0, ['currentlySmoking']] = 0

df.loc[df['smokingStatus'] == 1, ['neverSmoked']] = 0
df.loc[df['smokingStatus'] == 1, ['previouslySmoked']] = 1
df.loc[df['smokingStatus'] == 1, ['currentlySmoking']] = 0

df.loc[df['smokingStatus'] == 2, ['neverSmoked']] = 0
df.loc[df['smokingStatus'] == 2, ['previouslySmoked']] = 0
df.loc[df['smokingStatus'] == 2, ['currentlySmoking']] = 1

df.loc[df['smoking'] == 2, ['noOfCigarettesPerDay']] = 1
df.loc[df['smoking'] == 2, ['neverSmoked']] = 0
df.loc[df['smoking'] == 2, ['previouslySmoked']] = 0
df.loc[df['smoking'] == 2, ['currentlySmoking']] = 1

df.loc[df['alcoholFrequency'] == -3, ['alcoholFrequency']] = np.nan # -3 == won't say; not many occurences so we drop those

df.loc[df['diabetes'] == -1, ['diabetes']] = 0 # patient does not know if they have diabetes -> patient does not have it
df.loc[df['diabetes'] == -3, ['diabetes']] = np.nan # drop the few patients that won't tell if they have diabetes

df.loc[df['wheezeInChestInLastYear'] == -1, ['wheezeInChestInLastYear']] = 0
df.loc[df['wheezeInChestInLastYear'] == -3, ['wheezeInChestInLastYear']] = np.nan

df['systolicBloodPressure'] = df[['systolicBloodPressure0', 'systolicBloodPressure1']].mean(axis=1)
df['diastolicBloodPressure'] = df[['diastolicBloodPressure0', 'diastolicBloodPressure1']].mean(axis=1)

df = df.dropna(subset=['diastolicBloodPressure', 'weight', 'height', 'noOfCigarettesPerDay', 'alcoholFrequency', 'diabetes', 'wheezeInChestInLastYear', 'smoking', 'smokingStatus'])
df.drop(inplace=True, columns=['systolicBloodPressure0', 'systolicBloodPressure1', 'diastolicBloodPressure0', 'diastolicBloodPressure1', 'smoking', 'smokingStatus', 'stoppedSmokingFor6Months'])


y = df["sputumOnMostDays"]
X = df.drop(columns=["sputumOnMostDays"])
pfr = pandas_profiling.ProfileReport(df)
pfr.to_file("/tmp/df_report22_5.html")
df.describe()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1337)

#reg = linear_model.ElasticNet() #score: -1.395
#reg = linear_model.Ridge() #score: 0.27
#reg = linear_model.ElasticNetCV() #score: 0.27
#reg = linear_model.Lasso() #score: -1.395
#reg = svm.SVR() #0.098
#reg = GradientBoostingRegressor() #0.27
reg = svm.SVR(kernel='rbf')


reg.fit(X_train, y_train)
predictions = reg.predict(X_test)
print(reg.score(X_test, y_test))
difference = predictions - y_test
difference.sort_values(inplace=True)

#pca = PCA()
#pca.fit(X_train)
#transformed = pca.transform(X_train)
plt.figure()
plt.plot(range(0, 5000), difference[:5000])
#plt.plot(range(0, 100), predictions[100], label="predictions")
plt.savefig('/tmp/rbf2.png')
