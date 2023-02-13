import pandas as pd
import numpy as np
import math


class C45:
    def continuous_criterion(self, data_: pd.DataFrame, attribute, overall_entropy):
        data = data_.copy()
        best_gain = -math.inf
        values = sorted(data[attribute].unique())
        information = 0
        split_information = 0
        for i in range(len(values) - 1):
            threshold = (values[i] + values[i+1]) / 2

            check1 = data[data[attribute] < threshold]
            information += self.entropy(check1) * check1.shape[0]
            split_information += self.split_information(
                data.shape[0], check1.shape[0])

            check2 = data[data[attribute] > threshold]
            information += self.entropy(check2) * check2.shape[0]
            split_information += self.split_information(
                data.shape[0], check2.shape[0])

            gain = (overall_entropy - information) / split_information
            if gain > best_gain:
                best_gain = gain
                best_threshold = threshold

            information = 0
            split_information = 0

        return best_gain, best_threshold

    def discrete_criterion(self, data: pd.DataFrame, attribute, overall_entropy):
        values = data[attribute].unique()
        information = 0
        split_information = 0
        for value in values:
            check = data[data[attribute] == value]
            information += self.entropy(check) * check.shape[0]
            split_information += self.split_information(
                data.shape[0], check.shape[0])
        information = information / data.shape[0]
        if split_information == 0:
            gain = 0
        else:
            gain = (overall_entropy - information) / split_information
        return gain

    def split_information(self, N, n):
        return -(n/N) * math.log2(n/N)

    def entropy(self, data: pd.DataFrame):
        entropy = 0
        for i in self.classes:
            prob = ((data[data.columns[-1]] == i).sum()) / data.shape[0]
            if prob != 0:
                entropy += prob * math.log2(prob)
        entropy = entropy * -1
        return entropy

    def gain_criterion(self, data: pd.DataFrame):
        best_threshold = None
        best_gain = -math.inf
        overall_entropy = self.entropy(data)
        for attribute in data.columns[:-1]:
            # TODO: if multiple same, then array
            # for now assumed that this will not happen
            # or at least best is always singular best
            if check_discrete(data[attribute]):
                gain = self.discrete_criterion(
                    data, attribute, overall_entropy)
                new_threshold = None
            else:
                gain, new_threshold = self.continuous_criterion(
                    data, attribute, overall_entropy)

            if gain > best_gain:
                best_gain = gain
                best_attribute = attribute
                best_threshold = new_threshold

        return best_attribute, best_threshold

    def generate_tree(self, data: pd.DataFrame):
        y = data[data.columns[-1]]
        if is_single_class(y):
            pred = y.unique()[0]
            return Leaf(pred)
        split_attribute, threshold = self.gain_criterion(data)
        if threshold == None:
            categories = data[split_attribute].unique()
            node = CategoricalNode(split_attribute, categories)
            for cat in categories:
                new_data = data[data[split_attribute] == cat]
                node.children[cat] = self.generate_tree(new_data)
        else:
            node = ThresholdNode(split_attribute, threshold)
            node.children.append(self.generate_tree(
                data[data[split_attribute] < threshold]))
            node.children.append(self.generate_tree(
                data[data[split_attribute] > threshold]))
        return node

    def train(self, X: pd.DataFrame, y: pd.DataFrame):
        self.data = X.join(y)
        self.classes = y.unique()
        self.attributes = X.columns
        self.tree = self.generate_tree(self.data)

    def find_child(self, node, row):
        if type(node) == ThresholdNode:
            if row[node.attribute] < node.threshold:
                child_node = node.children[0]
            elif row[node.attribute] > node.threshold:
                child_node = node.children[1]
            else:
                return Leaf(None)
        elif type(node) == CategoricalNode:
            child_node = node.children[row[node.attribute]]
        else:
            print('ERROR: Can\'t find a child due to child type')
            exit()
        return child_node

    def iterate_tree(self, row):
        node = self.tree
        while True:
            if type(node) == Leaf:
                prediction = node.prediction
                break
            else:
                node = self.find_child(node, row)
        return prediction

    def predict(self, X: pd.DataFrame):
        predictions = list()
        for _, x in X.iterrows():
            prediction = self.iterate_tree(x)
            predictions.append(prediction)
        return predictions


class CategoricalNode:
    def __init__(self, attribute, categories):
        self.attribute = attribute
        self.categories = categories
        self.children = dict()


class ThresholdNode:
    def __init__(self, attribute, threshold):
        self.attribute = attribute
        self.threshold = threshold
        self.children = list()


class Leaf:
    def __init__(self, prediction):
        self.prediction = prediction
        self.children = list()


def is_single_class(y: pd.DataFrame):
    classes = len(y.unique())
    if classes > 1:
        return False
    else:
        return True


def check_discrete(column: pd.DataFrame):
    t = column.dtypes
    if t == 'object' or t == 'bool' or t == 'category':
        return True
    else:
        return False
