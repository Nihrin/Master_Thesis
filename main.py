import pandas as pd
import models.naive as naive
import models.tree as tree
import models.SPN as SPN
import helper_functions
from sklearn.model_selection import train_test_split
import os
import random
import pickle

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

def run_experiment(data: pd.DataFrame, filename: str, missing_data: list = [0], reproduction: bool = False):
    cross_validations = 30
    y = data['classes']
    y = pd.factorize(y)[0]
    X = data.drop(['classes'], axis=1)

    if reproduction:
        with open('C:/Users/s164389/Desktop/Afstuderen/Thesis/reproduction_random_states.pkl', 'rb') as f:
            random_dict = pickle.load(f)
        random_state = random_dict[filename]
        random.setstate(random_state)
    else:
        random_state = random.getstate()

    spn_distribution = helper_functions.get_spn_distributions()[filename]
    cspn_distribution = helper_functions.get_cspn_distributions()[filename]
    distributions = [spn_distribution, cspn_distribution]
    df = pd.DataFrame(columns=['%-missing', 'NBC acc', 'NCC low', 'NCC high', 'NCC robust', 'C4.5 acc', 'credal-C4.5 acc', 'SPN acc', 'CSPN low', 'CSPN high', 'CSPN robust'])

    for percentage in missing_data:
        for run in range(cross_validations):
            print('Missing', percentage, 'Run', run)
            random_state_int = random.randint(0, 20000)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=random_state_int)
            X_train = helper_functions.create_missing_data(X_train, percentage)
            NBC_acc, NCC_acc_low, NCC_acc_high, NCC_robust_acc = run_naive_classifiers(X_train, X_test, y_train, y_test)
            tree_acc, credal_tree_acc = run_tree_classifiers(X_train, X_test, y_train, y_test)
            SPN_acc, CSPN_acc_low, CSPN_acc_high, CSPN_robust_acc = run_SPN_classifiers(X_train, X_test, y_train, y_test, distributions, random_state_int)

            df.loc[len(df)] = [percentage, NBC_acc, NCC_acc_low, NCC_acc_high, NCC_robust_acc, tree_acc, credal_tree_acc, SPN_acc, CSPN_acc_low, CSPN_acc_high, CSPN_robust_acc]

    df = df.groupby(['%-missing']).mean()   
    print(filename)
    print(df)
    df.to_excel('C:/Users/s164389/Desktop/Afstuderen/Thesis/Results/' + filename[:-5] + '_results.xlsx')

    return random_state

def experiment(data_path, random_state_path, missing_data):
    col_names = helper_functions.get_names_dict()
    with open(random_state_path, 'rb') as f:
        random_dict = pickle.load(f)

    for filename in os.listdir(data_path):
        if filename in random_dict:
            continue
        print('Running', filename)
        data = pd.read_csv(data_path + filename, names=col_names[filename])
        random_state = run_experiment(data, filename, missing_data)
        random_dict[filename] = random_state
        with open(random_state_path, 'wb') as f:
            pickle.dump(random_dict, f)

data_path = 'C:/Users/s164389/Desktop/Afstuderen/Thesis/Data_Exp1/'
random_state_path = 'C:/Users/s164389/Desktop/Afstuderen/Thesis/reproduction_random_states.pkl'
missing_data = [0, 5, 10, 20, 30]

experiment(data_path, random_state_path, missing_data)