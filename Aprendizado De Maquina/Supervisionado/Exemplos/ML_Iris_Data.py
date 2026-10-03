import pandas as pd
import numpy as np

from sklearn.datasets import load_iris
iris_dataset = load_iris()

# Show species of Iris
print("Target names: {}".format(iris_dataset['target_names']))

# Show each feature 
print("Feature names: \n{}".format(iris_dataset['feature_names']))