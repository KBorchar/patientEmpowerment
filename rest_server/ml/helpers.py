def mongo2df(db, coll, limit=0):
    import pymongo
    import pandas as pd

    client = pymongo.MongoClient('localhost', 27017)  # connects to mongodb server
    db = client[f'{db}']  # select database on server ('use ukbb' in shell)
    collection = db[coll]
    df = pd.DataFrame(list(collection.find().limit(limit)))
    if '_id' in df.columns:
        df.drop(inplace=True, columns=["_id"])
    return df


def uuid():
    import uuid
    return uuid.uuid4().hex


def dump(things, names):
    from joblib import dump
    for i, t in enumerate(things):
        dump(t, f'data/models/{names[i]}.joblib')


def dump_JSON(columns):
    import json
    with open('data/columns.txt', 'w') as outfile:
        json.dump(columns, outfile)


def dump_config(df, imputer):
    import json
    import math

    columns = df.columns.format()
    minimums = df.min()
    maximums = df.max()
    f_config = {}
    for i, c in enumerate(columns):
        if imputer.initial_imputer_.statistics_[i] <= 1:
            f_config[c] = {"title": c,
                       "choices": {
                          "Yes": 1,
                          "No": 0
                        },
                       }
        else:
            f_config[c] = {"title": c,
                       "slider_min": math.floor(minimums[i]),
                       "slider_max": math.ceil(maximums[i])
                       }

    with open('data/features.txt', 'w') as outfile:
        json.dump(f_config, outfile)


def plot_classification_reports(reports, titles=[]):
    from matplotlib import colors
    import matplotlib.pyplot as plt
    import numpy as np

    ddl_heat = ['#DBDBDB', '#DCD5CC', '#DCCEBE', '#DDC8AF', '#DEC2A0', '#DEBB91',
                '#DFB583', '#DFAE74', '#E0A865', '#E1A256', '#E19B48', '#E29539']
    cmap = colors.ListedColormap(ddl_heat)

    for i, r in enumerate(reports):
        title = f'{titles[i]}-report' or 'Classification report'
        lines = r.split('\n')
        classes = []
        matrix = []

        for line in lines[2:(len(lines) - 5)]:
                s = line.split()
                classes.append(s[0])
                value = [float(x) for x in s[1: len(s) - 1]]
                matrix.append(value)

        fig, ax = plt.subplots(1)

        for column in range(len(matrix) + 1):
            for row in range(len(classes)):
                ax.text(column, row, matrix[row][column], va='center', ha='center')

        plt.imshow(matrix, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        x_tick_marks = np.arange(len(classes) + 1)
        y_tick_marks = np.arange(len(classes))
        plt.xticks(x_tick_marks, ['precision', 'recall', 'f1-score'], rotation=45)
        plt.yticks(y_tick_marks, classes)
        plt.ylabel('Classes')
        plt.xlabel('Measures')
        plt.savefig(f'/tmp/classification_reports/{titles[i]}_report_{uuid()}.png')
        plt.clf()


def generate_profile_report(df, dbname):
    import pandas_profiling
    pandas_profiling.ProfileReport(df).to_file(f"/tmp/profile_reports/r_{dbname}{uuid()}.html")


def get_args():
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("-o", "--output", action='store_true')
    parser.add_argument("-l", "--labels", dest="labels", default='COPD asthma diabetes tuberculosis',
                        help="provide labels to predict on, separated by spaces, e.g.: \"COPD asthma diabetes\"")
    parser.add_argument("-c", "--correlator", dest="correlator",
                        help="name the correlating feature that you want to superimpose onto the matplot plots")
    parser.add_argument("-db", "--database", dest="db", default='ukbb',
                        help="mongo collection to learn from"),
    parser.add_argument("-coll", "--collection", dest='collection', default='ahriMocked2',
                        help="collection name from the database")
    return parser.parse_args()


def own_predict(intercept, coef, X):
    import numpy as np
    return 1 - (1/(1+(np.exp(np.dot(coef, np.ravel(X.values)) + intercept))))

    #thetas = np.dot(coef, np.ravel(X.values)) + intercept
    #full = np.exp(thetas)
    #return 1/(1+full)