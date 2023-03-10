import pandas as pd
import math
import random
import pickle


def create_missing_data(data: pd.DataFrame, percentage):
    to_delete = math.ceil(data.shape[0] * data.shape[1] * (percentage / 100))
    cells = set()
    col_length = data.shape[0]
    row_length = data.shape[1]
    while len(cells) < to_delete:
        row = random.randint(0, row_length - 1)
        col = random.randint(0, col_length - 1)
        cells.add((row, col))
    
    for cell in cells:
        data.iat[cell[1], cell[0]] = None

    return data


def accuracies(y, classical_predictions, credal_predictions):
    classical_accuracy = (sum(1 for a, b in zip(classical_predictions, y)
                              if a == b) / len(y)) * 100
    # print('Classical accuracy: ' + str("%.2f" % classical_accuracy) + '%')

    credal_accuracy = (sum(1 for a, b in zip(credal_predictions, y)
                           if a == b) / len(y)) * 100
    # print('Credal accuracy: ' + str("%.2f" % credal_accuracy) + '%')

    d_correct = 0
    r_correct = 0
    r_count = 0
    for a, b in zip(credal_predictions, y):
        if a == b:
            d_correct += 1
            r_correct += 1
            r_count += 1
        elif b in a:
            d_correct += 1
        elif type(a) != list():
            r_count += 1
    discounted_credal_accuracy = (d_correct / len(y)) * 100
    robust_credal_accuracy = (r_correct / r_count) * 100

    # print('Discounted credal accuracy: ' + str("%.2f" %
    #       discounted_credal_accuracy) + '%')
    # print('Robust credal accuracy: ' + str("%.2f" %
    #       robust_credal_accuracy) + '%')

    return classical_accuracy, credal_accuracy, discounted_credal_accuracy, robust_credal_accuracy


def create_names_dict():
    dict = {
        'balance-scale': ['classes', 'left-weight', 'left-distance', 'right-weight', 'right-distance'],
        'iris': ['sepal_l', 'sepal_w', 'petal_l', 'petal_w', 'classes']      
    }
    with open('UCI_data_names.pkl', 'wb') as f:
        pickle.dump(dict, f)


def open_names_dict():
    with open('UCI_data_names.pkl', 'rb') as f:
        dict = pickle.load(f)
    return dict
