#lin_regs = [linear_model.ElasticNet(), linear_model.ElasticNetCV(), linear_model.Ridge(), linear_model.Lasso(), svm.SVR(), ensemble.GradientBoostingRegressor()]

#reg = linear_model.ElasticNet() #score: -1.395
#reg = linear_model.Ridge() #score: 0.27
#reg = linear_model.ElasticNetCV() #score: 0.27
#reg = linear_model.Lasso() #score: -1.395
#reg = svm.SVR() #0.098
#reg = GradientBoostingRegressor() #0.27


'''label_name = "COPD"
y = df[label_name]
X = df.drop(columns=[label_name])
pfr = pandas_profiling.ProfileReport(df)
pfr.to_file("/tmp/df_report22_5.html")
df.describe()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1337)

imputer = train_imputer(X_train)
rows = X_test.iloc[0:2]
rows2 = rows.copy()
rows.iloc[0, :] = np.nan
rows.iloc[1, -1:] = np.nan
result = imputer.transform(rows)

classies = [linear_model.LogisticRegressionCV(class_weight='balanced'), linear_model.LogisticRegression(class_weight='balanced')]

for i, c in enumerate(classies):
    print(c.__class__.__name__)
    c.fit(X_train, y_train)
    preds = c.predict(X_test)

    probabilities = c.predict_proba(X_test)
    probabilities = np.array(probabilities[:, 0])
    probabilities.sort()
    plt.figure()
    plt.plot(range(0, len(probabilities)), probabilities)
    plt.savefig('/tmp/probabilities' + label_name + c.__class__.__name__ + '.png')

    print(c.score(X_test, y_test))
    conf_mat.plot_confusion_matrix(y_test, preds, normalize=True)
    plt.savefig('/tmp/' + label_name + c.__class__.__name__ + str(i) + '.png')
'''


#predictions = c.predict(X_test)
#difference = predictions - y_test
#difference.sort_values(inplace=True)
#pca = PCA()
#pca.fit(X_train)
#transformed = pca.transform(X_train)
#plt.figure()
#plt.plot(range(0, 5000), difference[:5000])
#plt.plot(range(0, 100), predictions[100], label="predictions")
#plt.savefig('/tmp/rbf2.png')