### python:
from typing import List

from ml import io, analysis
from ml import learn

args = io.get_args()
df = io.mongo2df(args.db, args.collection)

if args.output:
    analysis.generate_profile_report(df, args.dbname)

labels = args.labels.split()
models, classification_reports = learn.train_models(df, labels, args.correlator)
analysis.plot_classification_reports(classification_reports, labels)
imputer = learn.train_imputer(df)

io.dump_models(models, labels)
io.dump_objects([imputer], ["imputer"])
io.dump_config(df, imputer)