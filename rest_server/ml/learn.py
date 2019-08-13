# Train models and simultaneously test them. Output some test visuals to file.

def train_models(df, labels=None, correlator=None):
    from sklearn import linear_model
    from sklearn.model_selection import train_test_split
    import matplotlib.pyplot as plt
    import numpy as np
    from ml import io, analysis
    from sklearn.metrics import classification_report
    import pandas as pd

    plt.figure()
    ax1 = plt.subplot()
    ax2 = ax1.twinx()
    models = []
    classification_reports = []

    if labels is None:
        labels = df.columns.format()

    for column in df.columns:
        if df[column].dtype == float:
            df = df.astype({column: 'int32'})

    for i, l in enumerate(labels):

        # Train each model, then add it to list of models. liblinear+balanced had best scores for UKBB data.
        y = df[l]
        X = df.drop(columns=[l])
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=7)
        model = linear_model.LogisticRegression(class_weight='balanced', solver='liblinear')
        models.append(model.fit(X_train, y_train))

        # Drawing
        # Probabilities for all entries in test set
        probabilities = model.predict_proba(X_test)

        # See if a correlator was set in input
        correlator_values = None
        try:
            correlator_values = X_test._series[correlator]._values
        except:
            correlator_values = []

        # sort all predicted probabilities from test set, then plot them. If correlator was set, draw it as well.
        if correlator_values == []:
            probabilities = np.array(probabilities[:, 0])
            probabilities.sort()
            plt.plot(range(0, len(probabilities)), probabilities, label=f'{l}')
        else:
            probabilities = np.array(probabilities[:, 0])
            mixed = list(zip(probabilities, correlator_values))
            mixed.sort(key=lambda x: x[0])
            probabilities_sorted = [m[0] for m in mixed]
            correlators_from_sorted = [m[1] for m in mixed]

            ax1.plot(range(0, len(probabilities_sorted)), probabilities_sorted, label=f'{l}')
            ax2.plot(range(0, len(correlators_from_sorted)), correlators_from_sorted, '--', linewidth=1,
                     color=(0.1, 0.2, 0.5, 0.1))

        # Will be relevant if -o is set form CLI.
        y_pred = model.predict(X_test)
        try:
            classification_reports.append(classification_report(y_test, y_pred, target_names=['no', 'yes']))
        except:
            pass

    plt.legend(loc='upper left')
    plt.savefig(f'/tmp/{models[i].__class__.__name__}{io.short_uuid()}.png')
    return models, classification_reports

# Experimental. Currently only used for keeping track of the means of each column.
# Imputes missing values by iteratively going through all columns and predicting on them. Unfortunately, broken when
# the data frame has no NaNs, see https://github.com/scikit-learn/scikit-learn/issues/13773
#
# Even though not used to potential now, this could have big implications on database compatibilities, as it would allow
# predictions on incomplete data sets.
def train_imputer(df):
    from sklearn import linear_model
    from sklearn.experimental import enable_iterative_imputer # NB: keep this import, else the Imputer won't work on the used sklearn version
    from sklearn.impute import IterativeImputer # if your IDE complains: Symptom of it being experimental. Still works.

    imputer = IterativeImputer(verbose=0,
                               estimator=linear_model.LogisticRegression(class_weight="balanced"),
                               sample_posterior=True,
                               n_nearest_features=5,
                               initial_strategy="mean")
    imputer.fit(df)
    return imputer