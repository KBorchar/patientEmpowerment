'''
def own_predict(intercept, coef, X):
    import numpy as np
    return 1 - (1/(1+(np.exp(np.dot(coef, np.ravel(X.values)) + intercept))))

    # thetas = np.dot(coef, np.ravel(X.values)) + intercept
    # full = np.exp(thetas)
    # return 1/(1+full)

'''