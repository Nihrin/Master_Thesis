import numpy as np
import pandas as pd
# import deeprob.spn.learning.learnspn as lspn
# import deeprob.spn.structure as spn

from cdeeprob.spn.learning import learnspn
from scipy.special import logsumexp

print(logsumexp(a=[[-1, -2],[-1,-2]],b=[[0,1],[1,0]], axis=1, keepdims=True))

# weights = np.array([0.35238096, 0.2952381, 0.35238096])
# lls = np.array([[-1.27634537e+02, 5.27358130e-02, -4.60587025e+00],[-8.34535956e-01, -3.12715569e+01, -5.35779495e+01]])
# print(np.log(np.sum(weights * np.exp(lls))))
# print(logsumexp(a=lls, b=weights, axis=1, keepdims=True))

# iris_names = ['sepal_l', 'sepal_w', 'petal_l', 'petal_w', 'classes']
# iris_data = pd.read_csv('UCI_data/iris.data', names=iris_names)
# iris_data['classes'] = iris_data['classes'].astype('category')
# iris_data['classes'] = iris_data['classes'].cat.codes
# data = iris_data.to_numpy()

# distributions = [spn.Gaussian] * 4 + [spn.Categorical]
# domains = [(4.3, 7.9), (2.0, 4.4), (1.0, 6.9), (0.1, 2.5), [0, 1, 2]]
# print(lspn.learn_spn(data, distributions, domains, min_rows_slice=50).children[0].children)


# FROM CSPN AND SPN PAPERS
# data = pd.read_csv('test_data/nltcs.test.data', header=None)
# print(data)







# domains = list()
# for i in range(len(data[0])):
#     domains.append((min(data[:, i]), max(data[:, i])))
# domains = [(4.3, 7.9), (2.0, 4.4), (1.0, 6.9), (0.1, 2.5), [0, 1], [0, 1], [0, 1]]
# distributions = [spn.Gaussian] * 4 + [spn.Bernoulli] * 3

# BALANCE SPN
# balance_names = ['classes', 'left-weight', 'left-distance', 'right-weight', 'right-distance']
# balance_data = pd.read_csv('UCI_data/balance-scale.data', names=balance_names)
# one_hot = pd.get_dummies(balance_data.classes, prefix='class')
# data = balance_data.drop(['classes'], axis=1)
# data = pd.concat([data, one_hot.reindex(data.index)], axis=1)
# data = data.to_numpy()

# print(data.shape)
# distributions = [spn.Categorical] * 4 + [spn.Bernoulli] * 3
# domains = [[1,2,3,4,5], [1,2,3,4,5], [1,2,3,4,5], [1,2,3,4,5], [0, 1], [0, 1], [0, 1]]
# print(lspn.learn_spn(data, distributions, domains).children[0].children[1].children[-1].children[0].children)

# balance_names = ['classes', 'left-weight', 'left-distance', 'right-weight', 'right-distance']
# balance_data = pd.read_csv('UCI_data/balance-scale.data', names=balance_names)
# balance_data['classes'] = balance_data['classes'].astype('category')
# balance_data['classes'] = balance_data['classes'].cat.codes
# data = balance_data.to_numpy()
# distributions = [spn.Categorical] * 5
# domains = [[0,1,2], [1,2,3,4,5], [1,2,3,4,5], [1,2,3,4,5], [1,2,3,4,5]]
# print(lspn.learn_spn(data, distributions, domains).children[1].children[0].children)


# GAUSSIAN MIXTURE
# from sklearn.mixture import GaussianMixture
# from sklearn.feature_selection import mutual_info_regression
# import math

# print(data)
# print(data.astype(np.float32))
# gm = GaussianMixture(n_components=2, random_state=0).fit(data)
# print(gm.means_)


# mi = mutual_info_regression(data, data[:,0])
# print(mi)


# GTEST DEEPROBKIT
# def gtest(
#     data: np.ndarray,
#     i: int,
#     j: int,
#     p: float = 5.0,
#     test: bool = True
# ):
#     """
#     The G-Test independence test between two features.
#     :param data: The data.
#     :param i: The index of the first feature.
#     :param j: The index of the second feature.
#     :param distributions: The distributions.
#     :param domains: The domains.
#     :param p: The threshold for the G-Test.
#     :param test: If the method is called as test (true) or as value of statistics (false), default True.
#     :return: False if the features are assumed to be dependent, True otherwise.
#     :raises ValueError: If the leaf distributions are discrete and continuous.
#     """
#     n_samples = len(data)
#     x1, x2 = data[:, i], data[:, j]
#     hist, _, _ = np.histogram2d(x1, x2, bins=math.ceil(n_samples**(1/2)))

#     # Compute G-test statistics
#     hist = hist.astype(np.float32) + np.finfo(np.float32).eps
#     m1 = np.sum(hist, axis=1, keepdims=True)
#     m2 = np.sum(hist, axis=0, keepdims=True)
#     e = m1 * m2 / n_samples
#     g_val = 2.0 * np.sum(hist * np.log(hist / e))

#     print(g_val)
#     # Return test result
#     if test:
#         f1 = np.count_nonzero(m1)
#         f2 = np.count_nonzero(m2)
#         dof = (f1 - 1) * (f2 - 1)
#         p_thresh = 2.0 * dof * p
#         print(p_thresh)
#         return g_val < p_thresh

#     # Return the value of G-test
#     return g_val

# g = gtest(X, 0, 3)
# print(g)





# FOR MEDIUM EXAMPLE EM (https://medium.com/@prateek.shubham.94/expectation-maximization-algorithm-7a4d1b65ca55)
# import numpy as np                              # import numpy
# from numpy.linalg import inv                    # for matrix inverse
# import matplotlib.pyplot as plt                 # import matplotlib.pyplot for plotting framework
# from scipy.stats import multivariate_normal     # for generating pdf
# import random



# m1 = [1,1]      # consider a random mean and covariance value
# m2 = [7,7]                                              
# cov1 = [[3, 2], [2, 3]]                                      
# cov2 = [[2, -1], [-1, 2]]
# x = np.random.multivariate_normal(m1, cov1, size=(200,))  # Generating 200 samples for each mean and covariance
# y = np.random.multivariate_normal(m2, cov2, size=(200,))
# d = iris_X[:,:4]

# m1 = random.choice(d)
# m2 = random.choice(d)
# cov1 = np.cov(np.transpose(d))
# cov2 = np.cov(np.transpose(d))
# pi = 0.5

# x1 = np.linspace(2,8,100)  
# x2 = np.linspace(2,8,100)
# X, Y = np.meshgrid(x1,x2) 

# Z1 = multivariate_normal(m1, cov1)  
# Z2 = multivariate_normal(m2, cov2)

# ##Expectation step
# def Estep(lis1):
#     m1=lis1[0]
#     m2=lis1[1]
#     cov1=lis1[2]
#     cov2=lis1[3]
#     pi=lis1[4]
    
#     pt2 = multivariate_normal.pdf(d, mean=m2, cov=cov2)
#     pt1 = multivariate_normal.pdf(d, mean=m1, cov=cov1)
#     w1 = pi * pt2
#     w2 = (1-pi) * pt1
#     eval1 = w1/(w1+w2)

#     return(eval1)


# ## Maximization step
# def Mstep(eval1):
#     num_mu1,din_mu1,num_mu2,din_mu2=0,0,0,0

#     for i in range(0,len(d)):
#         num_mu1 += (1-eval1[i]) * d[i]
#         din_mu1 += (1-eval1[i])

#         num_mu2 += eval1[i] * d[i]
#         din_mu2 += eval1[i]

#     mu1 = num_mu1/din_mu1
#     mu2 = num_mu2/din_mu2

#     num_s1,din_s1,num_s2,din_s2=0,0,0,0
#     for i in range(0,len(d)):

#         q1 = np.matrix(d[i]-mu1)
#         num_s1 += (1-eval1[i]) * np.dot(q1.T, q1)
#         din_s1 += (1-eval1[i])

#         q2 = np.matrix(d[i]-mu2)
#         num_s2 += eval1[i] * np.dot(q2.T, q2)
#         din_s2 += eval1[i]

#     s1 = num_s1/din_s1
#     s2 = num_s2/din_s2

#     pi = sum(eval1)/len(d)
    
#     lis2=[mu1,mu2,s1,s2,pi]
#     return(lis2)


# def plot(lis1):
#     mu1=lis1[0]
#     mu2=lis1[1]
#     s1=lis1[2]
#     s2=lis1[3]
#     Z1 = multivariate_normal(mu1[1:3], s1[1:3,1:3])  
#     Z2 = multivariate_normal(mu2[1:3], s2[1:3,1:3])

#     pos = np.empty(X.shape + (2,))                # a new array of given shape and type, without initializing entries
#     pos[:, :, 0] = X; pos[:, :, 1] = Y   

#     plt.figure(figsize=(5,5))                                                          # creating the figure and assigning the size
#     plt.scatter(d[:,0], d[:,1], marker='o')     
#     plt.contour(X, Y, Z1.pdf(pos), colors="r" ,alpha = 0.5) 
#     plt.contour(X, Y, Z2.pdf(pos), colors="b" ,alpha = 0.5) 
#     plt.axis('equal')                                                                  # making both the axis equal
#     plt.xlabel('X-Axis', fontsize=16)                                                  # X-Axis
#     plt.ylabel('Y-Axis', fontsize=16)                                                  # Y-Axis
#     plt.grid()                                                                         # displaying gridlines
#     plt.show()



# iterations = 100
# lis1=[m1,m2,cov1,cov2,pi]
# for i in range(0,iterations):
#     lis2 = Mstep(Estep(lis1))
#     lis1=lis2
#     if i % 10 == 9:
#         plot(lis1)
