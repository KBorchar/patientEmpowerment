def mongo2df(coll, limit=0):
    import pymongo
    import pandas as pd

    client = pymongo.MongoClient('localhost', 27017)  # connects to mongodb server
    db = client.ukbb  # select database on server ('use ukbb' in shell)
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
        dump(t, f'rest_server/data/models/{names[i]}.joblib')


def dumpJSON(columns):

    import json
    with open('rest_server/data/columns.txt', 'w') as outfile:
        json.dump(columns, outfile)


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


def parse_db_name(argv):
    if len(argv) == 1:
        return input("choose db to learn from")
    else:
        return argv[1]
