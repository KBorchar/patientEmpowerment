# pymongo
# http://api.mongodb.com/python/current/
# http://api.mongodb.com/python/current/tutorial.html

### python:
import helpers
import learn
import pandas_profiling

df = helpers.mongo2df('ahriCleaner')
pfr = pandas_profiling.ProfileReport(df)
pfr.to_file("/tmp/df_report_cleaned.html")
labels = ["COPD", "asthma", "diabetes", "tuberculosis"]
models = learn.train_models(df, labels)
imputer = learn.train_imputer(df)