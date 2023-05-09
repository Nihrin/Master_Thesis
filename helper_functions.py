import pandas as pd
import math
import random
import deeprob.spn.structure as spn
import cdeeprob.spn.structure as cspn


def create_missing_data(data: pd.DataFrame, percentage, random_state=None):
    if percentage <= 0:
        return data, random_state
    if random_state == None:
        random_state = random.getstate()
    cells = set()
    n_rows = data.shape[0]
    n_cols = data.shape[1]
    to_delete = math.ceil(n_rows * (percentage / 100))
    random.setstate(random_state)
    rows = random.sample(range(n_rows), to_delete)

    for row in rows:
        cols = random.choices(range(n_cols), k = n_cols-1)
        for col in cols:
            cells.add((row, col))
    
    for cell in cells:
        data.iat[cell[0], cell[1]] = None

    return data, random_state


def classical_accuracies(y_pred, y_test):
    accuracy = (sum(1 for a, b in zip(y_pred, y_test)
                              if a == b) / len(y_test)) * 100
    return accuracy

def credal_accuracies(y_pred, y_test):
    low_correct = 0
    up_correct = 0
    robust_count = 0
    for a, b in zip(y_pred, y_test):
        if type(a) is list:
            if b in a:
                up_correct += 1
        elif a == b:
            low_correct += 1
            up_correct += 1
            robust_count += 1
        else:
            robust_count += 1       

    low_acc = (low_correct / len(y_test)) * 100
    up_acc = (up_correct / len(y_test)) * 100
    robust_acc = (low_correct / robust_count) * 100

    return low_acc, up_acc, robust_acc

def get_names_dict():
    dict = {
        'balance-scale.data': ['classes', 'left-weight', 'left-distance', 'right-weight', 'right-distance'],
        'car.data': ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'classes'],
        'cmc.data': ['age', 'education_w', 'education_h', 'children', 'religion', 'working', 'occupation_h', 'solindex', 'media', 'classes'],
        'german-credit.data': ['status', 'duration', 'history', 'purpose', 'amount', 'savings', 'employment', 'installmentrate', 'personalstatus', 'debtors', 'residence', 'property', 'age', 'installmentplans', 'housing', 'existingcredits', 'job', 'numpeople', 'telephone', 'foreign', 'classes'],
        'iris.data': ['sepal_l', 'sepal_w', 'petal_l', 'petal_w', 'classes'],
        'wdbc.data': ['classes', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30'],
        'wine.data': ['classes', 'alcohol', 'malicacid', 'ash', 'alcalinityash', 'magnesium', 'phenols', 'flavanoids', 'nonflavphenols', 'proanthocyanins', 'colorintense', 'hue', 'diluted', 'proline']
    }
    return dict

def get_spn_distributions():
    dict = {
        'balance-scale.data': [spn.Categorical] * 4,
        'car.data': [spn.Categorical] * 6,
        'cmc.data': [spn.Gaussian, spn.Categorical, spn.Categorical, spn.Gaussian, spn.Categorical, spn.Categorical, spn.Categorical, spn.Gaussian, spn.Categorical],
        'german-credit.data': [],
        'iris.data': [spn.Gaussian] * 4,
        'wdbc.data': [],
        'wine.data': []
    }
    return dict

def get_cspn_distributions():
    dict = {
        'balance-scale.data': [cspn.Categorical] * 4,
        'car.data': [cspn.Categorical] * 6,
        'cmc.data':  [cspn.Gaussian, cspn.Categorical, cspn.Categorical, cspn.Gaussian, cspn.Categorical, cspn.Categorical, cspn.Categorical, cspn.Gaussian, cspn.Categorical],
        'german-credit.data': [],
        'iris.data': [cspn.Gaussian] * 4,
        'wdbc.data': [],
        'wine.data': []
    }
    return dict