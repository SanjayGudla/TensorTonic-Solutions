import numpy as np

def stable_sigmoid(x):
    # For positive values, use standard formula.
    # For negative values, rewrite to avoid overflow.
    return np.where(
        x >= 0, 
        1 / (1 + np.exp(-x)), 
        np.exp(x) / (1 + np.exp(x))
    )

def logistic_regression(X, y, lr=0.01, n_iters=1000):
    """
    Returns:
        tuple: (weights, bias) where weights is a list and bias is a float
    """
    X = np.array(X,dtype=float)
    y = np.array(y,dtype=float)

    num_points, point_dim = X.shape
    w = np.zeros(point_dim)
    b = 0

    for _ in range(n_iters):
        y_cap = stable_sigmoid(X@w+b)

        loss_weight_grad = (1/num_points)*(X.T @ (y_cap-y))
        loss_bias_grad = (1/num_points)*np.sum(y_cap-y)

        w = w - lr * loss_weight_grad
        b = b - lr * loss_bias_grad

    w = [float(v) for v in w]
    b = float(b)
    return w,b
