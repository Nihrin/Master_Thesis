import pandas as pd
import math
import random
import deeprob.spn.structure as spn
import cdeeprob.spn.structure as cspn


def create_missing_data(data: pd.DataFrame, percentage):
    if percentage <= 0:
        return data
    cells = set()
    n_rows = data.shape[0]
    n_cols = data.shape[1]
    to_delete = math.ceil(n_rows * (percentage / 100))
    rows = random.sample(range(n_rows), to_delete)

    for row in rows:
        cols = random.choices(range(n_cols), k = n_cols-1)
        for col in cols:
            cells.add((row, col))
    
    for cell in cells:
        data.iat[cell[0], cell[1]] = None

    return data


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
    if robust_count == 0:
        robust_acc = 0
    else:
        robust_acc = (low_correct / robust_count) * 100

    return low_acc, up_acc, robust_acc

def get_names_dict():
    dict = {
        'balance-scale.data': ['classes', 'left-weight', 'left-distance', 'right-weight', 'right-distance'],
        'car.data': ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'classes'],
        'cmc.data': ['age', 'education_w', 'education_h', 'children', 'religion', 'working', 'occupation_h', 'solindex', 'media', 'classes'],
        'dermatology.data': ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34', 'classes'],
        'german-credit.data': ['status', 'duration', 'history', 'purpose', 'amount', 'savings', 'employment', 'installmentrate', 'personalstatus', 'debtors', 'residence', 'property', 'age', 'installmentplans', 'housing', 'existingcredits', 'job', 'numpeople', 'telephone', 'foreign', 'classes'],
        'glass.data': ['1', '2', '3', '4', '5', '6', '7', '8', '9', 'classes'],
        'haberman.data': ['age', 'operations', 'nodes', 'classes'],
        'hepatitis.data': ['classes', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19'],
        'horse-colic.data': ['classes', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26'],
        'iris.data': ['sepal_l', 'sepal_w', 'petal_l', 'petal_w', 'classes'],
        'lymphography.data': ['classes', 'lymphatics', 'affere', 'lymph c', 'lymph s', 'by pass', 'extravasates', 'regen', 'uptake', 'nodes dim', 'nodes enl', 'changes l', 'defect', 'changes n', 'changes s', 'special', 'disloc', 'exlusion', 'no'],
        'nursery.data': ['1', '2', '3', '4', '5', '6', '7', 'classes'],
        'wdbc.data': ['classes', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30'],
        'wine.data': ['classes', 'alcohol', 'malicacid', 'ash', 'alcalinityash', 'magnesium', 'phenols', 'flavanoids', 'nonflavphenols', 'proanthocyanins', 'colorintense', 'hue', 'diluted', 'proline'],
        'zoo.data': ['hair', 'feathers', 'eggs', 'milk', 'airborne', 'aquatic', 'predator', 'toothed', 'backbone', 'breathes', 'venomous', 'fins', 'legs', 'tail', 'domestic', 'catsize', 'classes']  
    }
    return dict

def get_spn_distributions():
    dict = {
        'balance-scale.data': [spn.Categorical] * 4,
        'car.data': [spn.Categorical] * 6,
        'cmc.data': [spn.Gaussian, spn.Categorical, spn.Categorical, spn.Gaussian, spn.Categorical, spn.Categorical, spn.Categorical, spn.Gaussian, spn.Categorical],
        'dermatology.data': [spn.Categorical] * 33 + [spn.Gaussian],
        'german-credit.data': [spn.Categorical, spn.Gaussian, spn.Categorical, spn.Categorical, spn.Gaussian, spn.Categorical, spn.Categorical, spn.Gaussian, spn.Categorical, spn.Categorical, spn.Gaussian, spn.Categorical, spn.Gaussian, spn.Categorical, spn.Categorical, spn.Gaussian, spn.Categorical, spn.Gaussian, spn.Categorical, spn.Categorical],
        'glass.data': [spn.Gaussian] * 9,
        'haberman.data': [spn.Gaussian] * 3,
        'hepatitis.data': [spn.Gaussian, spn.Categorical, spn.Categorical, spn.Categorical, spn.Categorical, spn.Categorical, spn.Categorical, spn.Categorical, spn.Categorical, spn.Categorical, spn.Categorical, spn.Categorical, spn.Categorical, spn.Gaussian, spn.Gaussian, spn.Gaussian, spn.Gaussian, spn.Gaussian, spn.Categorical],
        'horse-colic.data': [spn.Categorical, spn.Gaussian, spn.Gaussian, spn.Gaussian, spn.Categorical, spn.Categorical, spn.Categorical, spn.Categorical, spn.Categorical, spn.Categorical, spn.Categorical, spn.Categorical, spn.Categorical, spn.Gaussian, spn.Categorical, spn.Categorical, spn.Gaussian, spn.Gaussian, spn.Categorical, spn.Gaussian, spn.Categorical, spn.Categorical, spn.Categorical, spn.Categorical, spn.Categorical, spn.Categorical],
        'iris.data': [spn.Gaussian] * 4,
        'lymphography.data': [spn.Categorical] * 18,
        'nursery.data': [spn.Categorical] * 7,
        'wdbc.data': [spn.Gaussian] * 30,
        'wine.data': [spn.Gaussian] * 13,
        'zoo.data': [spn.Categorical] * 16
    }
    return dict

def get_cspn_distributions():
    dict = {
        'balance-scale.data': [cspn.Categorical] * 4,
        'car.data': [cspn.Categorical] * 6,
        'cmc.data':  [cspn.Gaussian, cspn.Categorical, cspn.Categorical, cspn.Gaussian, cspn.Categorical, cspn.Categorical, cspn.Categorical, cspn.Gaussian, cspn.Categorical],
        'dermatology.data': [cspn.Categorical] * 33 + [cspn.Gaussian],
        'german-credit.data': [cspn.Categorical, cspn.Gaussian, cspn.Categorical, cspn.Categorical, cspn.Gaussian, cspn.Categorical, cspn.Categorical, cspn.Gaussian, cspn.Categorical, cspn.Categorical, cspn.Gaussian, cspn.Categorical, cspn.Gaussian, cspn.Categorical, cspn.Categorical, cspn.Gaussian, cspn.Categorical, cspn.Gaussian, cspn.Categorical, cspn.Categorical],
        'glass.data': [cspn.Gaussian] * 9,
        'haberman.data': [cspn.Gaussian] * 3,
                'hepatitis.data': [cspn.Gaussian, cspn.Categorical, cspn.Categorical, cspn.Categorical, cspn.Categorical, cspn.Categorical, cspn.Categorical, cspn.Categorical, cspn.Categorical, cspn.Categorical, cspn.Categorical, cspn.Categorical, cspn.Categorical, cspn.Gaussian, cspn.Gaussian, cspn.Gaussian, cspn.Gaussian, cspn.Gaussian, cspn.Categorical],
        'horse-colic.data': [cspn.Categorical, cspn.Gaussian, cspn.Gaussian, cspn.Gaussian, cspn.Categorical, cspn.Categorical, cspn.Categorical, cspn.Categorical, cspn.Categorical, cspn.Categorical, cspn.Categorical, cspn.Categorical, cspn.Categorical, cspn.Gaussian, cspn.Categorical, cspn.Categorical, cspn.Gaussian, cspn.Gaussian, cspn.Categorical, cspn.Gaussian, cspn.Categorical, cspn.Categorical, cspn.Categorical, cspn.Categorical, cspn.Categorical, cspn.Categorical],        
        'iris.data': [cspn.Gaussian] * 4,
        'lymphography.data': [cspn.Categorical] * 18,
        'nursery.data': [cspn.Categorical] * 7,
        'wdbc.data': [cspn.Gaussian] * 30,
        'wine.data': [cspn.Gaussian] * 13,
        'zoo.data': [cspn.Categorical] * 16
    }
    return dict