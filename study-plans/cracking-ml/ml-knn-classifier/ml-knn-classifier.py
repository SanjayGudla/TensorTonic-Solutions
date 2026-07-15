import numpy as np

def knn_classify(X_train, y_train, X_test, k):
    X_train = np.array(X_train, dtype=float)
    y_train = np.array(y_train, dtype=int)
    X_test = np.array(X_test, dtype=float)
    
    predictions = []
    
    for test_point in X_test:
        # Euclidean distance to every training point
        distances = np.sqrt(np.sum((X_train - test_point) ** 2, axis=1))
        
        # Indices of the k smallest distances (default argsort — NOT stable)
        nearest_idx = np.argsort(distances)[:k]
        nearest_labels = y_train[nearest_idx]
        
        # Count votes per label
        labels, counts = np.unique(nearest_labels, return_counts=True)
        max_count = counts.max()
        
        # Tie-break: smallest label among those with max count
        winners = labels[counts == max_count]
        predicted_label = int(winners.min())
        
        predictions.append(predicted_label)
    
    return predictions