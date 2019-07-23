# pymongo
# http://api.mongodb.com/python/current/
# http://api.mongodb.com/python/current/tutorial.html

### python:
from typing import List

from ml import helpers
from ml import learn

args = helpers.get_args()
df = helpers.mongo2df(args.db, args.collection)

if args.output:
    helpers.generate_profile_report(df, args.dbname)

labels = args.labels.split()
models, classification_reports = learn.train_models(df, labels, args.correlator)
helpers.plot_classification_reports(classification_reports, labels)
imputer = learn.train_imputer(df)

helpers.dump(models, labels)
helpers.dump([imputer], ["imputer"])
helpers.dump_JSON(df.columns.format())
helpers.dump_config(df, imputer)