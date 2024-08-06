# Task C, implement knn
from sklearn.base import BaseEstimator,ClassifierMixin
from scipy.spatial import distance as dist
from collections import Counter
import numpy as np
import pandas as pd

class kNN(BaseEstimator, ClassifierMixin):
    def __init__(self, n_neighbors:int = 3):
        self.n_neighbors = n_neighbors

    def fit(self, X, y):
        self.X_train = X
        self.y_train = np.array(y)
        return self

    def predict(self, X):
        #find distances for point X from each point in test set
        distances = dist.cdist(X, self.X_train)

        #find the k nearest neibours indexes
        neighbors_ind = np.argpartition(distances,kth= self.n_neighbors, axis=-1)[:,:self.n_neighbors]

        #find the lbals of closest neighbors
        neighbor_labels = self.y_train[neighbors_ind]

        # Determine the most common label among the nearest neighbors for each test point
        def most_common_label(labels):
            return Counter(labels).most_common(1)[0][0]

        # Apply the most_common_label function to each row of neighbor_labels
        predictions = np.apply_along_axis(most_common_label, 1, neighbor_labels)

        return predictions