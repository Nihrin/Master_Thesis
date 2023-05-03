import cdeeprob.spn.structure as cspn
from cdeeprob.spn.models.sklearn import CSPNClassifier

import deeprob.spn.structure as spn
from deeprob.spn.models.sklearn import SPNClassifier

def SPN(X_train, y_train, X_test, random_state):
    _, n_features = X_train.shape
    distributions = [spn.Gaussian] * (n_features)

    # Instantiate and fit a SPN classifier
    clf = SPNClassifier(
        distributions,
        learn_leaf='mle',     # Learn leaf distributions by MLE
        split_rows='gmm',     # Use K-Means for splitting rows
        split_cols='gvs',     # Use GVS for splitting columns
        min_rows_slice=15,    # The minimum number of rows required to split furthermore
        random_state=random_state,       # The random state, used for reproducibility
        verbose=False
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return y_pred
    
def CSPN(X_train, y_train, X_test, random_state):
    _, n_features = X_train.shape
    distributions = [cspn.Gaussian] * (n_features)

    # Instantiate and fit a SPN classifier
    clf = CSPNClassifier(
        distributions,
        learn_leaf='mle',     # Learn leaf distributions by MLE
        split_rows='gmm',     # Use K-Means for splitting rows
        split_cols='gvs',     # Use GVS for splitting columns
        min_rows_slice=15,    # The minimum number of rows required to split furthermore
        random_state=random_state,       # The random state, used for reproducibility
        verbose=False
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return y_pred