def train_models(df, labels):
    from sklearn import linear_model
    from sklearn.model_selection import train_test_split
    import matplotlib.pyplot as plt
    import numpy as np
    import helpers
    import conf_mat
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
        print(f'{l}score: {model.score(X_test, y_test)}')
        '''mocked: 
        COPDscore: 0.8302
        asthmascore: 0.8102
        diabetesscore: 0.7216
        tuberculosisscore: 0.5938
        
        cleaner: 
        COPDscore: 0.8443811132221518
        asthmascore: 0.8215115051720498
        diabetesscore: 0.7262331996340863
        tuberculosisscore: 0.60432059672085'''

    plt.legend(loc='upper left')
    plt.savefig(f'/tmp/{models[i].__class__.__name__}{helpers.uuid()}.png')
    #preds = m.predict(X_test)
    # print(m.score(X_test, y_test))
    #conf_mat.plot_confusion_matrix(y_test, preds, normalize=True)
    return models


def train_imputer(df):
    from sklearn import linear_model
    from sklearn.experimental import enable_iterative_imputer
    from sklearn.impute import IterativeImputer

    imputer = IterativeImputer(verbose=0,
                               estimator=linear_model.LogisticRegression(class_weight="balanced"),
                               sample_posterior=True,
                               n_nearest_features=5,
                               initial_strategy="mean")

    imputer.fit(df)
    return imputer