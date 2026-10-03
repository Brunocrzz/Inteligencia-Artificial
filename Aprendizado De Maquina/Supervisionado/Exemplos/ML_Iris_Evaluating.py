import pandas as pd
import numpy as np

from sklearn.datasets import load_iris
iris_dataset = load_iris()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
iris_dataset['data'], iris_dataset['target'], random_state=0)

# training and classification method
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)

# Training
knn.fit(X_train, y_train)

# Evaluating the Model
y_pred = knn.predict(X_test)

print("Test set score: {:.2f}".format(knn.score(X_test, y_test)))
'''
Same as:
    print("Test set score: {:.2f}".format(np.mean(y_pred == y_test)))
'''