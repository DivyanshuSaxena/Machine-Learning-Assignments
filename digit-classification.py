#!/usr/bin/env python
# coding: utf-8

# In[93]:


import math
import random
import numpy as np
from cvxopt import matrix
from cvxopt.solvers import qp

# Hyperparameters
gamma = 0.05
gaussian = False
digit = 6


# In[94]:


# Get data for PART A
train_data_raw = np.genfromtxt('./ass2_data/digit_train.csv', delimiter=',')
test_data_raw = np.genfromtxt('./ass2_data/digit_test.csv', delimiter=',')


# In[80]:


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


# In[86]:


get_class = lambda x : 1 if x == digit else -1

# Global Parameters for our model
# m = 500
m = len(train_data)
alphas = np.array([])
w = np.array([])
b = 0

P = np.zeros(shape=(m,m))
for i in range(m):
    for j in range(m):
        y_i = get_class(train_data[i][-1])
        y_j = get_class(train_data[j][-1])
        if gaussian:
            diff_vector = train_data[i][:-1] - train_data[j][:-1]
            kernel = math.exp(-gamma * np.dot(diff_vector.T, diff_vector))
            P[i][j] = y_i * y_j * kernel
        else:
            P[i][j] = y_i * y_j * np.dot(train_data[i][:-1], train_data[j][:-1])

# Get input for the solver
P = matrix(P)
q = matrix(1.0, (m,1))

G_identity = np.identity(m)
temp_G = np.concatenate((G_identity, -G_identity), axis=0)
G = matrix(temp_G, (2*m, m))

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


# In[90]:


# Evaluate w and b
w = np.zeros(28*28)
for i in range(m):
    sample = train_data[i]
    w += alphas[i] * get_class(sample[-1]) * sample[:-1]
# print (w)

maxone = -9999999
minone = 9999999
for sample in train_data[0:m]:
    y = get_class(sample[-1])
    if y == 1:
        minone = min(minone, np.dot(w.T, sample[:-1]))
    else:
        maxone = max(maxone, np.dot(w.T, sample[:-1]))

b = -(maxone + minone)/2
# print (b)


# In[92]:


# Find accuracy
accuracy = 0
test_m = len(test_data)

for sample in train_data[0:m]:
    pred_z = np.dot(w.T, sample[:-1]) + b
    if pred_z > 0:
        pred = 1
    else:
        pred = -1
    if pred == get_class(sample[-1]):
        accuracy += 1

print (accuracy/m * 100)


# In[ ]:




