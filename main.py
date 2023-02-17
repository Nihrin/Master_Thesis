import pandas as pd
import models.naive as naive
import models.tree as tree
import helper_functions
import math
from sklearn.model_selection import train_test_split


def run_naive_classifiers(X, y, missing):
    if missing > 0:
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


def run_tree_classifiers(X, y, missing):
    if missing > 0:
        X = helper_functions.create_missing_data(X, missing)
        X = X.fillna(X.mean())

    classical_tree = tree.CredalC45()
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
    s = 1
    classical_tree.train(X, y, s)
    # print(classical_tree.tree.threshold, classical_tree.tree.attribute)
    # exit()
    classical_predictions = classical_tree.predict(X)

    return classical_predictions, y


def tree_classifiers(X, y, missing: int = 0, runs: int = 10):
    pred = run_tree_classifiers(X, y, missing)
    return pred


def run():
    MISSING_DATA = 0
    CROSS_VALIDATION = 10

    # iris_names = ['sepal_l', 'sepal_w', 'petal_l', 'petal_w', 'classes']
    # iris_data = pd.read_csv('UCI_data/iris.data', names=iris_names)
    # iris_y = iris_data['classes']
    # iris_X = iris_data.drop(['classes'], axis=1)
    # pred, y_true = tree_classifiers(iris_X, iris_y, MISSING_DATA, CROSS_VALIDATION)
    # print(y_true.to_list())
    # print(pred)

    # balance_names = ['classes', 'left-weight', 'left-distance', 'right-weight', 'right-distance']
    # balance_data = pd.read_csv('UCI_data/balance-scale.data', names=balance_names)
    # balance_y = balance_data['classes']
    # balance_X = balance_data.drop(['classes'], axis=1)

    # print('Iris')
    # naive_classifiers(iris_X, iris_y, missing_data, cross_validations)
    # print()
    # print('Balance')
    # naive_classifiers(balance_X, balance_y, missing_data, cross_validations)

    # weather_names = ['Outlook', 'Temperature', 'Humidity', 'Wind', 'classes']
    # weather_data = pd.read_csv('test_data/weka_weather.data', names=weather_names)
    # weather_y = weather_data['classes']
    # weather_X = weather_data.drop(['classes'], axis=1)
    # pred = tree_classifiers(weather_X, weather_y,
    #                         MISSING_DATA, CROSS_VALIDATION)
    # print(weather_y.to_list())
    # print(pred)

    names = ['X1', 'X2', 'classes']
    data = pd.read_csv('test_data/paper_example.data', names=names)
    y = data['classes']
    X = data.drop(['classes'], axis=1)
    pred, y_true = tree_classifiers(X, y, MISSING_DATA, CROSS_VALIDATION)

    count = 0
    for a, b in zip(pred, y_true.to_list()):
        if a == b:
            count += 1
    print(count, 'out of', len(pred), 'correct')


# helper_functions.create_names_dict()
# names_dict = helper_functions.open_names_dict()
# print(names_dict)
run()
