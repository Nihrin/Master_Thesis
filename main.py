import pandas as pd
import naive
import helper_functions
import math


def run(missing):
    names = ['sepal_l', 'sepal_w', 'petal_l', 'petal_w', 'classes']
    data = pd.read_csv('data/iris.data', names=names)
    y = data['classes']
    X = data.drop(['classes'], axis=1)

    X = helper_functions.create_missing_data(X, missing)
    X = X.fillna(X.mean())

    NBC = naive.NBC()
    NBC.train(X, y)
    NBC_predictions = NBC.predict(X)

    NCC = naive.NCC()
    NCC.train(X, y)
    NCC_predictions = NCC.predict(X)

    NBC_acc, NCC_acc, NCC_d_acc = helper_functions.discounted_accuracies(
        y, NBC_predictions, NCC_predictions)

    # print('Mistakes:')
    # print('Index \t Prediction \t \t True')

    indices = []
    for i in range(len(NCC_predictions)):
        # if NCC_predictions[i] != y[i]:
        #     print(i+1, '\t', NCC_predictions[i], '\t', y[i])
        if type(NCC_predictions[i]) == list:
            indices.append(i)

    y_ = y.copy().to_list()

    for i in sorted(indices, reverse=True):
        del NCC_predictions[i]
        del NBC_predictions[i]
        del y_[i]

    NBC_e_acc, NCC_e_acc = helper_functions.exact_accuracies(
        y_, NCC_predictions, NBC_predictions)

    return NBC_acc, NCC_acc, NCC_d_acc, NBC_e_acc, NCC_e_acc


runs = 10
missing = 50
NBC_acc_, NCC_acc_, NCC_d_acc_, NBC_e_acc_, NCC_e_acc_ = list(
), list(), list(), list(), list()
for a in range(runs):
    NBC_acc, NCC_acc, NCC_d_acc, NBC_e_acc, NCC_e_acc = run(missing)
    NBC_acc_.append(NBC_acc)
    NCC_acc_.append(NCC_acc)
    NCC_d_acc_.append(NCC_d_acc)
    NBC_e_acc_.append(NBC_e_acc)
    NCC_e_acc_.append(NCC_e_acc)
    print()

print("Overview:")
print("NBC_acc: " + str("%.2f" % (math.fsum(NBC_acc_) / runs)) + '%')
print("NCC_acc: " + str("%.2f" % (math.fsum(NCC_acc_) / runs)) + '%')
print("NCC_d_acc: " + str("%.2f" % (math.fsum(NCC_d_acc_) / runs)) + '%')
print("NBC_e_acc: " + str("%.2f" % (math.fsum(NBC_e_acc_) / runs)) + '%')
print("NCC_e_acc: " + str("%.2f" % (math.fsum(NCC_e_acc_) / runs)) + '%')
