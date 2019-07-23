
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



'''
def own_predict(intercept, coef, X):
    import numpy as np
    return 1 - (1/(1+(np.exp(np.dot(coef, np.ravel(X.values)) + intercept))))

    # thetas = np.dot(coef, np.ravel(X.values)) + intercept
    # full = np.exp(thetas)
    # return 1/(1+full)

'''