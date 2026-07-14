import numpy as np

def softmax_regression(X, y, n_classes, lr=0.01, n_iters=1000):
    """
    Returns: tuple (weights, bias) where weights is a 2D list (d x K) and bias is a list of length K
    """
    X = np.array(X, dtype=float)
    y = np.array(y, dtype=float)

    num_points, point_dim = X.shape
    y_onehot = np.zeros((num_points, n_classes), dtype=int)
    for index,real_class in enumerate(y):
        y_onehot[index][int(real_class)]=1

    weights = np.zeros((point_dim,n_classes))
    bias = np.zeros(n_classes)

    for _ in range(n_iters):
        z = X@weights + bias
        z = np.exp(z)
        sum = np.sum(z)
        p = z / np.sum(z, axis=1, keepdims=True)
        diff = p - y_onehot

        weight_grad = (1/num_points)*(X.T @ diff)
        bias_grad = (1/num_points)*(np.ones(num_points).T @ diff)

        weights = weights - lr*weight_grad
        bias = bias - lr* bias_grad

    return weights.tolist(), bias.tolist()
        
