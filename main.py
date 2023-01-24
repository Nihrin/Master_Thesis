import pandas as pd
import naive
import helper_functions
import math


def run_naive_classifiers(X, y, missing):
    X = helper_functions.create_missing_data(X, missing)
    X = X.fillna(X.mean())

    NBC = naive.NBC()
    NBC.train(X, y)
    NBC_predictions = NBC.predict(X)

    NCC = naive.NCC()
    NCC.train(X, y)
    NCC_predictions = NCC.predict(X)

    NBC_acc, NCC_acc, NCC_d_acc, NCC_r_acc = helper_functions.accuracies(
        y, NBC_predictions, NCC_predictions)

    return NBC_acc, NCC_acc, NCC_d_acc, NCC_r_acc


def naive_classifiers(X, y, missing: int = 0, runs: int = 10):
    NBC_acc_list, NCC_acc_list, NCC_d_acc_list, NCC_r_acc_list = list(
    ), list(), list(), list()

    for _ in range(runs):
        NBC_acc, NCC_acc, NCC_d_acc, NCC_r_acc = run_naive_classifiers(
            X.copy(), y, missing)
        NBC_acc_list.append(NBC_acc)
        NCC_acc_list.append(NCC_acc)
        NCC_d_acc_list.append(NCC_d_acc)
        NCC_r_acc_list.append(NCC_r_acc)

    print("Overview:")
    print("NBC_acc: " + str("%.2f" % (math.fsum(NBC_acc_list) / runs)) + '%')
    print("NCC_acc: " + str("%.2f" % (math.fsum(NCC_acc_list) / runs)) + '%')
    print("NCC_d_acc: " + str("%.2f" % (math.fsum(NCC_d_acc_list) / runs)) + '%')
    print("NCC_r_acc: " + str("%.2f" % (math.fsum(NCC_r_acc_list) / runs)) + '%')


def run():
    iris_names = ['sepal_l', 'sepal_w', 'petal_l', 'petal_w', 'classes']
    iris_data = pd.read_csv('UCI_data/iris.data', names=iris_names)
    iris_y = iris_data['classes']
    iris_X = iris_data.drop(['classes'], axis=1)

    balance_names = ['classes', 'left-weight', 'left-distance', 'right-weight', 'right-distance']
    balance_data = pd.read_csv('UCI_data/balance-scale.data', names=balance_names)
    balance_y = balance_data['classes']
    balance_X = balance_data.drop(['classes'], axis=1)

    missing_data = 30
    cross_validations = 10

    print('Iris')
    naive_classifiers(iris_X, iris_y, missing_data, cross_validations)
    print()
    print('Balance')
    naive_classifiers(balance_X, balance_y, missing_data, cross_validations)

# helper_functions.create_names_dict()
# names_dict = helper_functions.open_names_dict()
# print(names_dict)
run()