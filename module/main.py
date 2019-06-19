# pymongo
# http://api.mongodb.com/python/current/
# http://api.mongodb.com/python/current/tutorial.html

### python:
import helpers
import learn
import pandas_profiling

dbname = input("choose db to learn from")
df = helpers.mongo2df(f'{dbname}')                          #old: ahriCleaner for 120k datapoints
pfr = pandas_profiling.ProfileReport(df)
pfr.to_file(f"/tmp/df_report_mocked{dbname}{helpers.uuid()}.html")
labels = ["COPD", "asthma", "diabetes", "tuberculosis"]
models = learn.train_models(df, labels)
imputer = learn.train_imputer(df)

helpers.dump(models, labels)
helpers.dump([imputer], ["imputer"])