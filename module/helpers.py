def mongo2df(coll, limit=0):
    import pymongo
    import pandas as pd

    client = pymongo.MongoClient('localhost', 27017)  # connects to mongodb server
    db = client.ukbb  # select database on server ('use ukbb' in shell)
    collection = db[coll]
    df = pd.DataFrame(list(collection.find().limit(limit)))
    df.drop(inplace=True, columns=["_id"])
    return df

def uuid():
    import uuid
    return uuid.uuid4().hex

def dump(things, names):
    from joblib import dump
    for i, t in enumerate(things):
        dump(t, f'rest_server/data/models/{names[i]}.joblib')