import pandas as pd
import math
import random
import numpy as np

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
        data.iat[cell[1], cell[0]] = np.nan

    return data

def discounted_accuracies(y, NBC_predictions, NCC_predictions):
    NBC_acc = (sum(1 for a,b in zip(NBC_predictions, y) if a == b) / len(y)) * 100
    print('NBC accuracy: ' + str("%.2f" % NBC_acc) + '%')

    NCC_acc = (sum(1 for a,b in zip(NCC_predictions, y) if a == b) / len(y)) * 100
    print('NCC accuracy: ' + str("%.2f" % NCC_acc) + '%')

    correct = 0
    for a,b in zip(NCC_predictions, y):
        if a == b:
            correct += 1
        elif b in a:
            correct += 1 / len(a)
    NCC_d_acc = (correct / len(y)) * 100
    print('NCC discounted accuracy: ' + str("%.2f" % NCC_d_acc) + '%')

    return NBC_acc, NCC_acc, NCC_d_acc

def exact_accuracies(y, NBC_predictions, NCC_predictions):
    NBC_e_acc = (sum(1 for a,b in zip(NBC_predictions, y) if a == b) / len(y)) * 100
    print('NBC robust accuracy: ' + str("%.2f" % NBC_e_acc) + '%')

    NCC_e_acc = (sum(1 for a,b in zip(NCC_predictions, y) if a == b) / len(y)) * 100
    print('NCC robust accuracy: ' + str("%.2f" % NCC_e_acc) + '%')

    return NBC_e_acc, NCC_e_acc