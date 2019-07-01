def train_models(df, labels, correlator=None):
    from sklearn import linear_model
    from sklearn.model_selection import train_test_split
    import matplotlib.pyplot as plt
    import numpy as np
    from ml import helpers
    from sklearn.metrics import classification_report

    plt.figure()
    ax1 = plt.subplot()
    ax2 = ax1.twinx()
    models = []
    classification_reports = []

    for i, l in enumerate(labels):

        #Training and adding to list of models
        y = df[l]
        X = df.drop(columns=[l])
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=7)
        model = linear_model.LogisticRegression(class_weight='balanced', solver='liblinear')
        models.append(model.fit(X_train, y_train))

        #skl = model.predict_proba(X_test.head(1))
        #own = helpers.own_predict(models[i].intercept_[0], models[i].coef_, X_test.head(1))

        #Drawing
        probabilities = model.predict_proba(X_test)

        shadow_values = None
        try:
            shadow_values = X_test._series[correlator]._values
        except:
            shadow_values = []

        if shadow_values == []: #truth value bugs out otherwise
            probabilities = np.array(probabilities[:, 0])
            probabilities.sort()
            plt.plot(range(0, len(probabilities)), probabilities, label=f'{l}')
        else:
            probabilities = np.array(probabilities[:, 0])
            mixed = list(zip(probabilities, shadow_values))
            mixed.sort(key=lambda x: x[0])
            probabilities_sorted = [m[0] for m in mixed]
            correlators_from_sorted = [m[1] for m in mixed]

            ax1.plot(range(0, len(probabilities_sorted)), probabilities_sorted, label=f'{l}')
            ax2.plot(range(0, len(correlators_from_sorted)), correlators_from_sorted, '--', linewidth=1,
                     color=(0.1, 0.2, 0.5, 0.1))

        y_pred = model.predict(X_test)
        classification_reports.append(classification_report(y_test, y_pred, target_names=['no', 'yes']))

        #print(f'{l}score: {model.score(X_test, y_test)}')

    plt.legend(loc='upper left')
    plt.savefig(f'/tmp/{models[i].__class__.__name__}{helpers.uuid()}.png')
    return models, classification_reports


def train_imputer(df):
    from sklearn import linear_model
    from sklearn.experimental import enable_iterative_imputer
    from sklearn.impute import IterativeImputer

    imputer = IterativeImputer(verbose=0,
                               estimator=linear_model.LogisticRegression(class_weight="balanced"),
                               n_nearest_features=5,
                               initial_strategy="mean",
                               max_iter=1)
    imputer.fit(df)
    return imputer