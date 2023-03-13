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

    def train(self, X: pd.DataFrame, y: pd.DataFrame):
        self.X_train = X
        self.y_train = y
        self.classes = np.unique(y)

        self.set_likelihood()
        self.set_priors()

    def get_likelihood(self, values: list, c):
        prob = 1
        for i in range(len(self.X_train.columns)):
            col_name = self.X_train.columns[i]
            value = values[i]
            prob = prob * self.likelihood[col_name][value][c]
        return prob

    def get_prior(self, c):
        return self.priors[c]

    def get_marginal(self, values: list, c):
        classes = list(self.classes)
        classes.remove(c)
        prior = 1
        for x in classes:
            prior = prior * self.priors[x]
        likelihood = 1
        for i in range(len(values)):
            p = 0
            col_name = self.X_train.columns[i]
            value = values[i]
            for x in classes:
                p += self.likelihood[col_name][value][x]
            likelihood = likelihood * p
        marginal = prior * likelihood
        return marginal

    def bayes(self, l, p, m):
        post = l * p / ((l * p) + m)
        return post

    def predict(self, X: pd.DataFrame):
        predictions = list()
        for _, x in X.iterrows():
            posteriors = list()
            for c in self.classes:
                likelihood = self.get_likelihood(x, c)
                prior = self.get_prior(c)
                marginal = self.get_marginal(x, c)
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
                    self.likelihood[col_name][value][c] = count / total_a

    def set_priors(self):
        for c in self.classes:
            X_c = self.X_train[self.y_train == c]
            self.priors[c] = X_c.shape[0] / self.X_train.shape[0]

    def train(self, X: pd.DataFrame, y: pd.DataFrame, s):
        self.X_train = X
        self.y_train = y
        self.s = s
        self.classes = np.unique(y)

        self.set_likelihood()
        self.set_priors()

    def get_likelihood(self, values: list, c):
        prob = 1
        for i in range(len(self.X_train.columns)):
            col_name = self.X_train.columns[i]
            value = values[i]
            prob = prob * self.likelihood[col_name][value][c]
        return prob

    def get_prior(self, c):
        return self.priors[c]

    def get_marginal(self, values: list, c):
        classes = list(self.classes)
        classes.remove(c)
        prior = 1
        for x in classes:
            prior = prior * self.priors[x]
        likelihood = 1
        for i in range(len(values)):
            p = 0
            col_name = self.X_train.columns[i]
            value = values[i]
            for x in classes:
                p += self.likelihood[col_name][value][x]
            likelihood = likelihood * p
        marginal = prior * likelihood
        return marginal

    def bayes(self, l, p, m):
        post = l * p / ((l * p) + m)
        lower = post * (self.X_train.shape[0] / (self.X_train.shape[0] + self.s))
        upper = post * (self.X_train.shape[0] / (self.X_train.shape[0] + self.s)) + (self.s / (self.X_train.shape[0] + self.s))
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
                marginal = self.get_marginal(x, c)
                prob = self.bayes(likelihood, prior, marginal)
                posteriors.append(prob)
            prediction = self.interval_dominance(posteriors)
            predictions.append(prediction)
        return predictions
