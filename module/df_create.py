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

cursor = collection.find().limit(100000)  # TODO: dont limit in production
df = pd.DataFrame(list(cursor))
df.replace("NA", np.nan, inplace=True)
df = df.dropna(subset=["COPD"])
#df.drop(inplace=True, columns = ["_id", "patientId"])

df["neverSmoked"] = np.nan
df["previouslySmoked"] = np.nan
df["currentlySmoking"] = np.nan

df.loc[np.isnan(df['noOfCigarettesPreviouslyPerDay']), ['noOfCigarettesPreviouslyPerDay']] = 0
df.loc[np.isnan(df['noOfCigarettesPerDay']), ['noOfCigarettesPerDay']] = 0
df.loc[df['noOfCigarettesPerDay'] < 0, ['noOfCigarettesPerDay']] = np.nan

for index, row in df.iterrows():
    df["systolicBloodPressure0"] = (row["systolicBloodPressure0"] + row["systolicBloodPressure1"]) / 2
    df["diastolicBloodPressure0"] = (row["diastolicBloodPressure0"] + row["diastolicBloodPressure1"]) / 2

    mask = (df["noOfCigarettesPreviouslyPerDay"] == np.nan)
    df.loc[mask, "noOfCigarettesPreviouslyPerDay"] = 0

    if df["noOfCigarettesPreviouslyPerDay"][index] == np.nan:
        df["noOfCigarettesPreviouslyPerDay"][index] = 0

    if df["noOfCigarettesPerDay"][index] == np.nan:
        df["noOfCigarettesPerDay"][index] = 0
    elif df["noOfCigarettesPerDay"][index] < 0:
        df["noOfCigarettesPerDay"][index] = np.nan

    if df["smokingStatus"][index] == 0:
        df["neverSmoked"][index] = 1
        df["previouslySmoked"][index] = 0
        df["currentlySmoking"][index] = 0
    elif df["smokingStatus"][index] == 1:
        df["neverSmoked"][index] = 0
        df["previouslySmoked"][index] = 1
        df["currentlySmoking"][index] = 0
    elif df["smokingStatus"][index] == 2:
        df["neverSmoked"][index] = 0
        df["previouslySmoked"][index] = 0
        df["currentlySmoking"][index] = 1

    if df["smoking"][index] == 2:
        df["noOfCigarettesPerDay"][index] = 1
        df["neverSmoked"][index] = 0
        df["previouslySmoked"][index] = 0
        df["currentlySmoking"][index] = 1


df.drop(inplace=True, columns=["systolicBloodPressure1", "diastolicBloodPressure1"])
df.rename(index=str, inplace=True, columns={"systolicBloodPressure0": "systolicBloodPressure", "diastolicBloodPressure0": "diastolicBloodPressure"})
df = df.dropna(subset=["diastolicBloodPressure", "weight", "height", "noOfCigarettesPerDay", "smoking", "smokingStatus"])


y = df["sputumOnMostDays"]
X = df.drop(columns=["sputumOnMostDays"])
pfr = pandas_profiling.ProfileReport(df)
pfr.to_file("/tmp/df_report22_5.html")
df.describe()
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.25, random_state=1337)

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
plt.plot(range(0, 6000), difference[:6000])
#plt.plot(range(0, 100), predictions[100], label="predictions")
plt.savefig('/tmp/rbf.png')

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
        # pdb.set_trace()
        counts = column.value_counts(normalize=True, dropna=False)
        # pdb.set_trace()
        # if counts[0] is list:

        if "NA" in counts:
            print(counts["NA"])
            NA_percentages[column_index] = counts["NA"]

    return


#analyze_NAs(df)
