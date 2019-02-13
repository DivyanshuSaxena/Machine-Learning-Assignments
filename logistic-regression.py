#!/usr/bin/env python
# coding: utf-8

# In[149]:


import numpy as np
import matplotlib.pyplot as plt
import math


# In[150]:


# Read data from files
x_i_raw = np.genfromtxt('./ass1_data/logisticX.csv', delimiter=',')
y_i = np.genfromtxt('./ass1_data/logisticY.csv', delimiter=',')
theta = np.array([0.00, 0.00, 0.00])
m = x_i_raw.shape[0]
n = 1.5

# Normalize data - Both dimensions are normalized separately
mean1 = 0
mean2 = 0
squared_sum1 = 0
squared_sum2 = 0
for xi1, xi2 in x_i_raw:
    mean1 += xi1
    mean2 += xi2
    squared_sum1 += xi1*xi1
    squared_sum2 += xi2*xi2
mean1 = mean1/m
mean2 = mean2/m
e_x_squared1 = squared_sum1/m
e_x_squared2 = squared_sum2/m
variance1 = e_x_squared1 - mean1*mean1
variance2 = e_x_squared2 - mean2*mean2

x_i_norm = np.array([[(xi1-mean1)/math.sqrt(variance1), (xi2-mean2)/math.sqrt(variance2)]
                     for xi1, xi2 in x_i_raw])

normalize = False
if normalize:
    x_i = np.array([[1, xi1, xi2] for xi1, xi2 in x_i_norm])
else:
    x_i = np.array([[1, xi1, xi2] for xi1, xi2 in x_i_raw])


# In[151]:


# Detect Convergence
epsilon = 0.000000001
def converged(theta_next, theta):
    converged = True
    for d in range(theta.size):
        converged = converged and abs(theta_next[d] - theta[d]) < epsilon
    return converged


# In[152]:


# Implement Newton's Method for PART A
num_iterations = 0

while(True):
    # print (theta)
    del_l = np.array([0.00, 0.00, 0.00])
    hessian = np.zeros(shape=(theta.size, theta.size))
    for i in range(m):
        hyp = 1/(1 + math.exp(np.dot(theta, x_i[i])))
        del_l += (y_i[i] - hyp)*x_i[i]
        for d1 in range(theta.size):
            for d2 in range(theta.size):
                hessian[d1][d2] += hyp*(1-hyp)*x_i[i][d1]*x_i[i][d2]
    hessian_inv = np.linalg.inv(hessian)
    theta_next = theta - np.dot(hessian_inv, del_l)
    
    if (converged(theta_next, theta)):
        break
        
    theta = theta_next
    num_iterations += 1

print (theta)
print (num_iterations)


# In[153]:


# Plot the graph of logistic regression for PART B
label_0 = []
label_1 = []
plot_raw = True

for i in range(m):
    prediction = 1/(1 + math.exp(np.dot(theta, x_i[i])))
    # print ((x_i_raw[i], prediction))
    if (prediction >= 0.5):
        if plot_raw:
            label_1.append([x_i_raw[i][0], x_i_raw[i][1]])
        else:
            label_1.append([x_i[i][1], x_i[i][2]])
    else:
        if plot_raw:
            label_0.append([x_i_raw[i][0], x_i_raw[i][1]])
        else:
            label_0.append([x_i[i][1], x_i[i][2]])

label_0 = np.array(label_0)
label_1 = np.array(label_1)
plt.scatter(label_0[:, [0]], label_0[:, [1]], 50, c='r', marker='+')
plt.scatter(label_1[:, [0]], label_1[:, [1]], 50, c='b', marker='.')

# Plot the separator
if plot_raw:
    x1 = np.linspace(2, 8, 100)
else:
    x1 = np.linspace(-3, 3, 100)    
x2 = -(theta[0] + theta[1]*x1)/theta[2]
plt.plot(x1, x2, 'g-')


# In[ ]:




