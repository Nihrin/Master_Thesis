import numpy as np
import pandas as pd


class NBC:
    def __init__(self):
        self.priors = dict()  # Dict with class: prob
        # Dict with column: dict, with dict value: dict, with dict class: prob
        self.likelihood = dict()
        self.marginal = dict()  # Dict with column: dict, with dict value: prob

    def set_likelihood(self):
        for col_name in self.X_train.columns:
            column = self.X_train[col_name]
            self.likelihood[col_name] = dict()
            for value in np.unique(column):
                total_a = column.value_counts()[value]
                self.likelihood[col_name][value] = dict()
                for c in self.classes:
                    total_a_given_c = column[self.y_train == c]
                    try:
                        count = total_a_given_c.value_counts()[value]
                    except:
                        count = 0
                    self.likelihood[col_name][value][c] = count / total_a

    def set_priors(self):
        for c in self.classes:
            X_c = self.X_train[self.y_train == c]
            self.priors[c] = X_c.shape[0] / self.X_train.shape[0]

    def set_marginal(self):
        for col_name in self.X_train.columns:
            column = self.X_train[col_name]
            self.marginal[col_name] = dict()
            for value in np.unique(column):
                count = column.value_counts()[value]
                total = column.shape[0]
                self.marginal[col_name][value] = count / total

    def train(self, X: pd.DataFrame, y: pd.DataFrame):
        self.X_train = X
        self.y_train = y
        self.classes = np.unique(y)

        self.set_likelihood()
        self.set_priors()
        self.set_marginal()

    def get_likelihood(self, values: list, c):
        prob = 1
        for i in range(len(self.X_train.columns)):
            col_name = self.X_train.columns[i]
            value = values[i]
            prob = prob * self.likelihood[col_name][value][c]
        return prob

    def get_prior(self, c):
        return self.priors[c]

    def get_marginal(self, values: list):
        prob = 0
        for i in range(len(self.X_train.columns)):
            col_name = self.X_train.columns[i]
            value = values[i]
            prob += self.marginal[col_name][value]
        return prob

    def bayes(self, l, p, m):
        post = l * p / m
        return post

    def predict(self, X: pd.DataFrame):
        predictions = list()
        for _, x in X.iterrows():
            posteriors = list()
            for c in self.classes:
                likelihood = self.get_likelihood(x, c)
                prior = self.get_prior(c)
                marginal = self.get_marginal(x)
                prob = self.bayes(likelihood, prior, marginal)
                posteriors.append(prob)
            prediction = self.classes[np.argmax(posteriors)]
            predictions.append(prediction)
        return predictions


class NCC:
    def __init__(self):
        self.priors = dict()  # Dict with class: prob
        # Dict with column: dict, with dict value: dict, with dict class: prob
        self.likelihood = dict()
        self.marginal = dict()  # Dict with column: dict, with dict value: prob

    def set_likelihood(self):
        for col_name in self.X_train.columns:
            column = self.X_train[col_name]
            self.likelihood[col_name] = dict()
            for value in np.unique(column):
                total_a = column.value_counts()[value]
                self.likelihood[col_name][value] = dict()
                for c in self.classes:
                    total_a_given_c = column[self.y_train == c]
                    try:
                        count = total_a_given_c.value_counts()[value]
                    except:
                        count = 0
                    lower = count / (total_a + self.s)
                    upper = (count + self.s) / (total_a + self.s)
                    self.likelihood[col_name][value][c] = (lower, upper)

    def set_priors(self):
        for c in self.classes:
            X_c = self.X_train[self.y_train == c]
            lower = X_c.shape[0] / (self.X_train.shape[0] + self.s)
            upper = (X_c.shape[0] + self.s) / (self.X_train.shape[0] + self.s)
            self.priors[c] = (lower, upper)

    def set_marginal(self):
        for col_name in self.X_train.columns:
            column = self.X_train[col_name]
            self.marginal[col_name] = dict()
            for value in np.unique(column):
                count = column.value_counts()[value]
                total = column.shape[0]
                lower = count / (total + self.s)
                upper = (count + self.s) / (total + self.s)
                self.marginal[col_name][value] = (lower, upper)

    def train(self, X: pd.DataFrame, y: pd.DataFrame, s: int = 1):
        self.X_train = X
        self.y_train = y
        self.s = s
        self.classes = np.unique(y)

        self.set_likelihood()
        self.set_priors()
        self.set_marginal()

    def get_likelihood(self, values: list, c):
        lower = 1
        upper = 1
        for i in range(len(self.X_train.columns)):
            col_name = self.X_train.columns[i]
            value = values[i]
            lower = lower * self.likelihood[col_name][value][c][0]
            upper = upper * self.likelihood[col_name][value][c][1]
        return (lower, upper)

    def get_prior(self, c):
        return self.priors[c]

    def get_marginal(self, values: list):
        lower = 0
        upper = 0
        for i in range(len(self.X_train.columns)):
            col_name = self.X_train.columns[i]
            value = values[i]
            lower += self.marginal[col_name][value][0]
            upper += self.marginal[col_name][value][1]
        return (lower, upper)

    def bayes(self, l, p, m):
        lower = l[0] * p[0] / m[1]
        upper = l[1] * p[1] / m[0]
        return (lower, upper)

    def interval_dominance(self, intervals: list):
        best_indices = list(range(len(intervals)))
        for i in range(len(intervals)):
            for j in range(len(intervals)):
                if i != j:
                    if intervals[i][1] <= intervals[j][0]:
                        best_indices.remove(i)
                        break
        if len(best_indices) == 1:
            prediction = self.classes[best_indices[0]]
        elif len(best_indices) > 1:
            prediction = []
            for i in best_indices:
                prediction.append(self.classes[i])
        else:
            prediction = ''
        return prediction

    def predict(self, X: pd.DataFrame):
        predictions = list()
        for _, x in X.iterrows():
            posteriors = list()
            for c in self.classes:
                likelihood = self.get_likelihood(x, c)
                prior = self.get_prior(c)
                marginal = self.get_marginal(x)
                prob = self.bayes(likelihood, prior, marginal)
                posteriors.append(prob)
            prediction = self.interval_dominance(posteriors)
            predictions.append(prediction)
        return predictions
