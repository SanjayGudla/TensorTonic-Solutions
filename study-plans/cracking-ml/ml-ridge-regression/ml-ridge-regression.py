def ridge_regression(X, y, lr, epochs, alpha):
    """
    Perform ridge regression using gradient descent.
    Returns: tuple of (weights_list, bias)
    """
    X = np.array(X, dtype=float)
    y = np.array(y, dtype=float)

    num_points, point_dim = X.shape
    w = np.zeros(point_dim)
    b = 0

    for _ in range(epochs):
        y_cap = X@w + b

        weight_grad = (2/num_points) * (X.T@(y_cap-y))+(2*alpha*w)
        bias_grad = (2/num_points) * np.sum(y_cap-y)

        w = w - (lr * weight_grad)
        b = b - (lr * bias_grad)

    return w.tolist(), b
    
    

    