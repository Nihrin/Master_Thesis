import numpy as np
import pandas as pd

class NBC:
    def __init__(self):
        self.priors = dict() # Dict with class: prob
        self.likelihood = dict() # Dict with column: dict, with dict value: dict, with dict class: prob 
        self.marginal = dict() # Dict with column: dict, with dict value: prob 

    def set_likelihood(self):
        for c in self.classes:
            X_c = self.X_train[self.y_train == c]
            self.likelihood[c] = dict()
            group = X_c.groupby(X_c.columns.tolist(),as_index=False).size()
            for _, i in group.iterrows():
                self.likelihood[c][tuple(i[:-1].tolist())] = i[-1] / self.X_train.shape[0]

    def set_priors(self):
        for c in self.classes:
            X_c = self.X_train[self.y_train == c]
            self.priors[c] = X_c.shape[0] / self.X_train.shape[0]

    def set_marginal(self):
        group = self.X_train.groupby(self.X_train.columns.tolist(),as_index=False).size()
        for _, i in group.iterrows():
            if i[-1] > 1:
                print(i[:-1].tolist(), i[-1])
            self.marginal[tuple(i[:-1].tolist())] = i[-1] / self.X_train.shape[0]
        exit()
        
    def train(self, X, y):
        self.X_train = X
        self.y_train = y
        self.classes = np.unique(y)

        self.set_likelihood()
        self.set_priors()
        self.set_marginal()    

    def get_likelihood(self, x, c):
        if c in self.likelihood and x in self.likelihood[c]:
            return self.likelihood[c][x]
        else:
            return 0
    
    def get_prior(self, c):
        if c in self.priors:
            return self.priors[c]
        else:
            return 0
    
    def get_marginal(self, x):
        if x in self.marginal:
            return self.marginal[x]
        else:
            return 0

    def bayes(self, l, p, m):
        post = l * p / m
        print(l, p, m)
        return post

    def predict(self, X):
        predictions = list()
        for _, r in X.iterrows():
            x = tuple(r)
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
        self.priors = dict() # Dict with class: prob
        self.likelihood = dict() # Dict with column: dict, with dict value: dict, with dict class: prob 
        self.marginal = dict() # Dict with column: dict, with dict value: prob 

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

    def train(self, X, y):
        self.X_train = X
        self.y_train = y
        self.classes = np.unique(y)

        self.set_likelihood()
        self.set_priors()
        self.set_marginal()    

    def get_likelihood(self, values: list, c):
        prob = 1
        for i in range(len(values)):
            col_name = self.X_train.columns[i]
            value = values[i]
            prob = prob * self.likelihood[col_name][value][c]
        return prob
    
    def get_prior(self, c):
        return self.priors[c]
    
    def get_marginal(self, values: list):
        prob = 1
        for i in range(len(values)):
            col_name = self.X_train.columns[i]
            value = values[i]
            prob = prob * self.marginal[col_name][value]
        return prob       

    def IDM_bayes(self, l, p, m, s):
        N = self.X_train.shape[0]
        z = s / (N + s)
        lower_p = (l * p) / (m + z)
        upper_p = ((l * p) + z) / (m + z)
        print((l * p), m, z)
        post = (lower_p, upper_p)
        return post
    
    def classify(self, intervals: list):
        return prediction

    def predict(self, X, s: int = 1):
        predictions = list()
        for _, x in X.iterrows():
            posteriors = list()
            for c in self.classes:
                likelihood = self.get_likelihood(x, c)
                prior = self.get_prior(c)
                marginal = self.get_marginal(x)
                prob = self.IDM_bayes(likelihood, prior, marginal, s)
                posteriors.append(prob)
            print(posteriors)
            exit()
            prediction = self.classify(posteriors)
            predictions.append(prediction)
        return predictions