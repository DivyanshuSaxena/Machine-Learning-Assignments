#!/usr/bin/env python
# coding: utf-8

# In[27]:


import math
import random
import numpy as np
from cvxopt import matrix
from cvxopt.solvers import qp
import svmutil

# Hyperparameters
gamma = 0.05
gaussian = True
digit = 6


# In[28]:


# Get data for PART A
train_data_raw = np.genfromtxt('./ass2_data/digit_train.csv', delimiter=',')
test_data_raw = np.genfromtxt('./ass2_data/digit_test.csv', delimiter=',')


# In[29]:


# Process data to get the relevant vectors
scale_down = lambda x : x/255
train_data = np.array([np.zeros(shape=28*28 + 1)])
test_data = np.array([np.zeros(shape=28*28 + 1)])

for sample in train_data_raw:
    if sample[-1] == digit or sample[-1] == digit+1:
        insert_sample = np.array([scale_down(x) for x in sample[:-1]])
        insert_sample = np.append(insert_sample, sample[-1])
        train_data = np.vstack([train_data, insert_sample])
        
for sample in test_data_raw:
    if sample[-1] == digit or sample[-1] == digit+1:
        insert_sample = np.array([scale_down(x) for x in sample[:-1]])
        insert_sample = np.append(insert_sample, sample[-1])
        test_data = np.vstack([test_data, insert_sample])

train_data = train_data[1:]
test_data = test_data[1:]


# In[ ]:


get_class = lambda x : 1 if x == digit else -1

# Global Parameters for our model
# m = 500
m = len(train_data)
alphas = np.array([])
w = np.array([])
b = 0

# Get input for the solver
X = np.delete(train_data, -1, axis=1)
if not gaussian:
    Y = np.diag([get_class(y) for y in train_data[:, -1]])
    kernel = np.matmul(X, X.T)
    temp_P = np.matmul(np.matmul(Y, kernel), Y)
    P = matrix(temp_P)
else:
    Y = np.diag([get_class(y) for y in train_data[:, -1]])
    xtx = np.sum(np.multiply(X, X), 1).reshape(m, 1)
    kernel_noexp = xtx + xtx.T - 2 * np.dot(X, X.T)
    kernel = np.power(np.exp(-1*gamma), kernel_noexp)
    temp_P = np.matmul(np.matmul(Y, kernel), Y)
    P = matrix(temp_P)
    
q = matrix(1.0, (m,1))

G = matrix(np.identity(m))
G_identity = np.identity(m)
temp_G = np.concatenate((G_identity, -G_identity), axis=0)
G = matrix(temp_G)
    
h = matrix(0.0, (m,1))
h_zero = np.zeros(m)
h_ones = np.ones(m)
temp_h = np.append(h_zero, h_ones)
h = matrix(temp_h, (2*m,1))

temp_A = list(map(get_class, train_data[0:m, -1]))
A = matrix(np.array(temp_A), (1, m), 'd')

b = matrix(0.0)

# Use the cvxopt solver qp module
alphas = qp(P, q, G, h, A, b)['x']
alphas = np.array(-alphas)[:, 0]
print (alphas)
print (len(alphas))


# In[ ]:


# Evaluate w if linear kernel used
if not gaussian:
    w = np.zeros(28*28)
    for i in range(m):
        sample = train_data[i]
        w += alphas[i] * get_class(sample[-1]) * sample[:-1]
    # print (w)

# Evaluate b
# Find support vectors
support_vectors = []
epsilon = 0.0001
for i in range(m):
    if alphas[i] > epsilon:
        support_vectors.append(i)
print (len(support_vectors))

alpha_y = np.multiply(alphas, np.array(list(map(get_class, train_data[:, -1]))) )

w_trans_X = np.matmul(kernel, alpha_y)
maxone = -99999999
minone = 99999999
for i in range(m):
    wtx = w_trans_X[i]
    y = get_class(train_data[i][-1])
    if y == -1:
        maxone = max(maxone, wtx)
    else:
        minone = min(minone, wtx)

b = -(maxone + minone)/2
            
print (b)


# In[ ]:


# Find accuracy
accuracy = 0
test_m = len(test_data)

for sample in test_data[0:test_m]:
    if not gaussian:
        pred_z = np.dot(w.T, sample[:-1]) + b
        if pred_z > 0:
            pred = 1
        else:
            pred = -1
        if pred == get_class(sample[-1]):
            accuracy += 1
    else:
        xtx = np.sum(np.multiply(sample[:-1], sample[:-1])).reshape(1,1)
        XtX = np.sum(np.multiply(X, X), 1).reshape(m, 1)
        inner_product = xtx + XtX.T - 2 * np.dot(sample[:-1], X.T)
        wtx = np.dot(alpha_y, np.power(np.exp(-gamma), inner_product.T))
        pred_z = wtx + b
        if pred_z > 0:
            pred = 1
        else:
            pred = -1
        if pred == get_class(sample[-1]):
            accuracy += 1

print (accuracy/test_m * 100)


# In[30]:


# Use libsvm
Y = train_data[:, -1]
X = np.delete(train_data, -1, axis=1)
m_linear = svmutil.svm_train(Y, X, "-t 0 -c 1")
m_gaussian = svmutil.svm_train(Y, X, "-t 2 -c 1 -g 0.05")
print (m_gaussian.get_nr_sv())
print (m_linear.get_nr_sv())


# In[31]:


Y_test = test_data[:, -1]
X_test = np.delete(test_data, -1, axis=1)
labels_linear = svmutil.svm_predict(Y_test, X_test, m_linear)
labels_gaussian = svmutil.svm_predict(Y_test, X_test, m_gaussian)


# In[ ]:




