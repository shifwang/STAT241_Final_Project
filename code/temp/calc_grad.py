import numpy as np

theta = np.asarray([0, 1]) # first element is mu, second is sigma^2
data = np.reshape(np.random.randn(1000), (-1, 1))
data = np.array([.5, 1])
data.shape

K = 100
d = 2
candidates = np.random.choice(1000, K, replace = True)


def calc_grad(data, theta, candidates = np.array(range(0, data.shape[0]))):
    mu = theta[0]
    sigma2 = theta[1]
    out1 = (data[candidates,:] - mu) / sigma2 * 2
    out2 = out1**2/4 - 1/sigma2

    grads = np.column_stack((out1, out2))
    return grads

calc_grad(data, theta, candidates)