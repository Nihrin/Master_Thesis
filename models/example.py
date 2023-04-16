import numpy as np
from sklearn.datasets import load_iris
from sklearn import mixture
from sklearn.model_selection import train_test_split
import random
import math

import cdeeprob.spn.structure as spn
from cdeeprob.spn.models.sklearn import SPNClassifier

def create_missing_data(data: np.array, percentage):
    # to_delete = math.ceil(len(data) * len(data[0]) * (percentage / 100))
    # cells = set()
    # col_length = len(data)
    # row_length = len(data[0])
    # while len(cells) < to_delete:
    #     row = random.randint(0, row_length - 1)
    #     col = random.randint(0, col_length - 1)
    #     cells.add((row, col))
    
    cells = {(2, 36), (0, 48), (1, 16), (1, 49), (3, 13), (2, 51), (3, 22), (2, 11), (0, 11), (1, 21), (2, 87), (0, 93), (2, 90), (1, 94), (0, 41), (1, 36), (3, 82), (3, 79), (0, 62), (2, 7), (2, 10), (1, 75), (1, 14), (2, 22), (1, 93), (3, 11), (0, 52), (0, 49), (2, 52), (3, 78), (2, 61), (1, 65), (0, 6), (3, 29), (2, 64), (0, 70), (1, 10), (3, 96), (3, 102), (1, 80), (0, 82), (1, 22)}
    for cell in cells:
        data[cell[1], cell[0]] = None

    return data

if __name__ == '__main__':
    # Load the dataset and set the features distributions
    data, target = load_iris(return_X_y=True)
    _, n_features = data.shape
    distributions = [spn.Gaussian] * (n_features)
    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.30, random_state=42)
    missing = 10

    X_train = create_missing_data(X_train, missing)

    # =================================================================================================================

    # Instantiate and fit a SPN classifier
    clf = SPNClassifier(
        distributions,
        learn_leaf='mle',     # Learn leaf distributions by MLE
        split_rows='gmm',  # Use K-Means for splitting rows
        split_cols='gvs',     # Use RDC for splitting columns
        min_rows_slice=15,    # The minimum number of rows required to split furthermore
        random_state=42,       # The random state, used for reproducibility
        verbose=False
    )
    clf.fit(X_train, y_train)

    # Compute the accuracy score
    print('Train data -- Accuracy: {:.2f}'.format(clf.score(X_train, y_train)))

    # Sample some data from the conditional distribution and compute the accuracy score
    print('Test data -- Accuracy: {:.2f}'.format(clf.score(X_test, y_test)))