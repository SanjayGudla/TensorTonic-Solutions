import numpy as np

def linear_regression(X, y, lr, epochs):
    X = np.array(X, dtype=float)
    y = np.array(y, dtype=float)
    num_points, data_dimension = X.shape
    w = np.zeros(data_dimension)
    b = 0.0

    for _ in range(epochs):
        y_cap = X@w + b
        error = y_cap - y
        
        loss_weight_grad = (2/num_points)* (X.T @ error)
        bias_weight_grad = (2/num_points)* np.sum(error)
        w = w - lr * (loss_weight_grad)
        b = b - lr * (bias_weight_grad)

    weights = [round(float(value),4) for value in w]
    bias = round(float(b),4)
    return (weights, bias)
        
        

    
