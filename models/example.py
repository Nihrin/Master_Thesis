import numpy as np
from sklearn.datasets import load_iris
from sklearn import mixture
from sklearn.model_selection import train_test_split
import random
import math

import cdeeprob.spn.structure as spn
from cdeeprob.spn.models.sklearn import SPNClassifier

def create_missing_data(data: np.array, percentage):
    to_delete = math.ceil(len(data) * len(data[0]) * (percentage / 100))
    cells = set()
    col_length = len(data)
    row_length = len(data[0])
    while len(cells) < to_delete:
        row = random.randint(0, row_length - 1)
        col = random.randint(0, col_length - 1)
        cells.add((row, col))
    
    for cell in cells:
        data[cell[1], cell[0]] = None

    return data

if __name__ == '__main__':
    # Load the dataset and set the features distributions
    data, target = load_iris(return_X_y=True)
    _, n_features = data.shape
    distributions = [spn.Gaussian] * (n_features)
    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.30, random_state=42)

    X_train = create_missing_data(X_train, 30)

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
    print(clf.predict(X_test))
    print(y_test)
    # Compute the accuracy score
    # print('Train data -- Accuracy: {:.2f}'.format(clf.score(X_train, y_train)))

    # Sample some data from the conditional distribution and compute the accuracy score
    print('Test data -- Accuracy: {:.2f}'.format(clf.score(X_test, y_test)))