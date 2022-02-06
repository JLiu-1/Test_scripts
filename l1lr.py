import numpy as np 
from scipy.optimize import minimize
def fit(X, params):
    return X.dot(params)


def cost_function(params, X, y):
    return np.sum(np.abs(y - fit(X, params)))

X = np.asarray([np.ones((100,)), np.arange(0, 100)],dtype=np.double).T
print(X.shape)
y = 10 + 5 * np.arange(0, 100) + 25 * np.random.random((100,))

output = minimize(cost_function, x0=np.array([10.,5.],dtype=np.double).T, args=(X, y))
print(output)