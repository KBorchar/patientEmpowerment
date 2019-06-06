# pymongo
# http://api.mongodb.com/python/current/
# http://api.mongodb.com/python/current/tutorial.html

### python:
import helpers
import learn

df = helpers.mongo2df('ahriCleaner')
labels = ["COPD", "asthma", "diabetes", "tuberculosis"]
models = learn.train_models(df, labels)
imputer = learn.train_imputer(df)

