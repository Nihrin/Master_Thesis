import pandas as pd
import models.naive as naive
import models.tree as tree
import models.SPN as SPN
import helper_functions
from sklearn.model_selection import train_test_split
import os
import random

import warnings
warnings.filterwarnings("ignore")

def run_naive_classifiers(X_train, X_test, y_train, y_test):
    X_train = X_train.fillna(X_train.mean())

    NBC = naive.NBC()
    NBC.train(X_train, y_train)
    NBC_predictions = NBC.predict(X_test)

    NBC_acc = helper_functions.classical_accuracies(NBC_predictions, y_test)

    s = 2
    NCC = naive.NCC()
    NCC.train(X_train, y_train, s)
    NCC_predictions = NCC.predict(X_test)

    NCC_acc_low, NCC_acc_high, NCC_robust_acc = helper_functions.credal_accuracies(NCC_predictions, y_test)

    return NBC_acc, NCC_acc_low, NCC_acc_high, NCC_robust_acc

def run_tree_classifiers(X_train, X_test, y_train, y_test):
    classical_tree = tree.C45()
    classical_tree.train(X_train, y_train)
    classical_predictions = classical_tree.predict(X_test)
    classical_acc = helper_functions.classical_accuracies(classical_predictions, y_test)

    credal_tree = tree.CredalC45()
    credal_tree.train(X_train, y_train, 1)
    credal_predictions = credal_tree.predict(X_test)
    credal_acc = helper_functions.classical_accuracies(credal_predictions, y_test)

    return classical_acc, credal_acc

def run_SPN_classifiers(X_train, X_test, y_train, y_test, distributions, random_state):
    X_test_num = X_test.to_numpy()
    X_train_num = X_train.to_numpy()

    credal_predictions = SPN.CSPN(X_train_num, X_test_num, y_train, distributions[1], random_state)
    credal_acc_low, credal_acc_high, credal_robust_acc = helper_functions.credal_accuracies(credal_predictions, y_test)

    X_train = X_train.fillna(X_train.mean())
    X_train_num = X_train.to_numpy()
    classical_predictions = SPN.SPN(X_train_num, X_test_num, y_train, distributions[0], random_state)
    classical_acc = helper_functions.classical_accuracies(classical_predictions, y_test)

    return classical_acc, credal_acc_low, credal_acc_high, credal_robust_acc

def run_experiment1(data: pd.DataFrame, filename: str):
    missing_data = [0, 5, 10, 20, 30]
    cross_validations = 30
    y = data['classes']
    y = pd.factorize(y)[0]
    X = data.drop(['classes'], axis=1)

    reproduction_dict = dict()
    spn_distribution = helper_functions.get_spn_distributions()[filename]
    cspn_distribution = helper_functions.get_cspn_distributions()[filename]
    distributions = [spn_distribution, cspn_distribution]
    naive_df = pd.DataFrame(columns=['%-missing', 'classic-acc', 'credal-low', 'credal-high', 'credal-robust'])
    tree_df = pd.DataFrame(columns=['%-missing', 'classic-acc', 'credal-acc'])
    spn_df = pd.DataFrame(columns=['%-missing', 'classic-acc', 'credal-low', 'credal-high', 'credal-robust'])
    for percentage in missing_data:
        for run in range(cross_validations):
            print('Missing', percentage, 'Run', run)
            random_state_int = random.randint(0, 20000)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=random_state_int)
            X_train, random_state_missing = helper_functions.create_missing_data(X_train, percentage)
            NBC_acc, NCC_acc_low, NCC_acc_high, NCC_robust_acc = run_naive_classifiers(X_train, X_test, y_train, y_test)
            tree_acc, credal_tree_acc = run_tree_classifiers(X_train, X_test, y_train, y_test)
            SPN_acc, CSPN_acc_low, CSPN_acc_high, CSPN_robust_acc = run_SPN_classifiers(X_train, X_test, y_train, y_test, distributions, random_state_int)

            naive_df.loc[len(naive_df)] = [percentage, NBC_acc, NCC_acc_low, NCC_acc_high, NCC_robust_acc]
            tree_df.loc[len(tree_df)] = [percentage, tree_acc, credal_tree_acc]
            spn_df.loc[len(spn_df)] = [percentage, SPN_acc, CSPN_acc_low, CSPN_acc_high, CSPN_robust_acc]

            # reproduction_dict[(filename, missing_data, run)] = (random_state_int, random_state_missing)
            
    print('Naive')
    print(naive_df.groupby(['%-missing']).mean())
    print('C4.5')
    print(tree_df.groupby(['%-missing']).mean())
    print('SPN')
    print(spn_df.groupby(['%-missing']).mean())


def experiment1():
    abs_path = 'C:/Users/s164389/Desktop/Afstuderen/Thesis/Data_Exp1/'
    col_names = helper_functions.get_names_dict()

    for filename in os.listdir(abs_path):
        filename = 'cmc.data'
        print('Running', filename)
        data = pd.read_csv(abs_path + filename, names=col_names[filename])
        run_experiment1(data, filename)
        exit()

# data = pd.read_csv('C:/Users/s164389/Desktop/Afstuderen/Thesis/Data_touse/cmc.data', names=helper_functions.get_names_dict()['cmc.data'])
# data = data.astype({'age': 'float', 'children': 'float'})
# data.to_csv('cmc.data', index=False)
# exit()

experiment1()