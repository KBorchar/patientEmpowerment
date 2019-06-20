# pymongo
# http://api.mongodb.com/python/current/
# http://api.mongodb.com/python/current/tutorial.html

### python:
import helpers
import learn
import pandas_profiling
import sys

dbname = helpers.parse_db_name(sys.argv)
df = helpers.mongo2df(f'{dbname}')
pfr = pandas_profiling.ProfileReport(df)
pfr.to_file(f"/tmp/df_report_{dbname}{helpers.uuid()}.html")

labels = ["COPD", "asthma", "diabetes", "tuberculosis"]
models, classification_reports = learn.train_models(df, labels)
helpers.plot_classification_reports(classification_reports, labels)
imputer = learn.train_imputer(df)

helpers.dump(models, labels)
helpers.dump([imputer], ["imputer"])
helpers.dumpJSON(df.columns.format())