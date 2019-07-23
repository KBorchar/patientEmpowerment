from ml import io

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
        plt.savefig(f'/tmp/classification_report_{titles[i]}{io.short_uuid()}.png')
        plt.clf()


def generate_profile_report(df, dbname):
    import pandas_profiling
    pandas_profiling.ProfileReport(df).to_file(f"/tmp/profile_report_{dbname}{io.short_uuid()}.html")

