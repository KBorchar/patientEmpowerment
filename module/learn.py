def train_models(df, labels):
    from sklearn import linear_model
    from sklearn.model_selection import train_test_split
    import matplotlib.pyplot as plt
    import numpy as np

    plt.figure()
    models = []

    for i, l in enumerate(labels):
        y = df[l]
        X = df.drop(columns=[l])
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=7)
        model = linear_model.LogisticRegression(class_weight='balanced')
        models.append(model.fit(X_train, y_train))

        probabilities = model.predict_proba(X_test)
        probabilities = np.array(probabilities[:, 0])
        probabilities.sort()
        plt.plot(range(0, len(probabilities)), probabilities, label=f'{l}')

        # preds = m.predict(X_test)
        # print(m.score(X_test, y_test))
        # conf_mat.plot_confusion_matrix(y_test, preds, normalize=True)

    plt.legend(loc='upper left')
    plt.savefig('/tmp/' + models[i].__class__.__name__ + '.png')
    return models


def train_imputer(df):
    from sklearn import linear_model
    from sklearn.experimental import enable_iterative_imputer
    from sklearn.impute import IterativeImputer

    imputer = IterativeImputer(verbose=2,
                               estimator=linear_model.LogisticRegression(class_weight="balanced"),
                               sample_posterior=True,
                               n_nearest_features=5,
                               initial_strategy="most_frequent")

    imputer.fit(df)
    return imputer

