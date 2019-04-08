#!/usr/bin/env python
# coding: utf-8



import numpy as np
import matplotlib.pyplot as plt
import math
import sys



# Read data from files
x_i = np.genfromtxt(sys.argv[1])
outputs = open(sys.argv[2], 'r')
y_i_raw = outputs.read().split('\n')
theta = np.array([0.00, 0.00, 0.00])
m = x_i.shape[0]
type = int(sys.argv[3])

def get_label(y):
    if (y == 'Alaska'):
        return 0
    return 1

y_i = [get_label(y) for y in y_i_raw]




# Calculate means for PART A
mu_0 = np.array([0.00, 0.00])
mu_1 = np.array([0.00, 0.00])
num_labels_0 = 0
num_labels_1 = 0

for i in range(m):
    if (y_i[i] == 0):
        mu_0 += x_i[i]
        num_labels_0 += 1
    else:
        mu_1 += x_i[i]
        num_labels_1 += 1
        
mu_0 = mu_0 / num_labels_0
mu_1 = mu_1 / num_labels_1
print (mu_0, mu_1)




# Calculate Covariance Matrices
sigma = np.zeros(shape=(2,2))
sigma_0 = np.zeros(shape=(2,2))
sigma_1 = np.zeros(shape=(2,2))
phi = 0
for i in range(m):
    if (y_i[i] == 0):
        x_mu = np.array([x_i[i] - mu_0])
        sigma_0 += np.dot(np.transpose(x_mu), x_mu)
    else:
        x_mu = np.array([x_i[i] - mu_1])
        sigma_1 += np.dot(np.transpose(x_mu), x_mu)
        phi += 1
    sigma += np.dot(np.transpose(x_mu), x_mu)
    
sigma = sigma / m
sigma_0 = sigma_0 / num_labels_0
sigma_1 = sigma_1 / num_labels_1
phi = phi/m

print (sigma)
print (phi)




x_i_0, x_i_1 = [], []

for i in range(m):
    if (y_i[i] == 0):
        x_i_0.append(x_i[i])
    else:
        x_i_1.append(x_i[i])
x_i_0 = np.array(x_i_0)
x_i_1 = np.array(x_i_1)

if type is 0:
    # Plot training data for PART B
    plt.scatter(x_i_0[:, [0]], x_i_0[:, [1]], marker='+')
    plt.scatter(x_i_1[:, [0]], x_i_1[:, [1]], marker='*')

    # Evaluate function between x1 and x2 to plot separator for PART C
    sigma_inv = np.linalg.inv(sigma)
    coeff = 2 * (np.dot(mu_0.T, sigma_inv) - np.dot(mu_1.T, sigma_inv))
    intercept_1 = np.dot( np.dot(mu_1.T, sigma_inv), mu_1 )
    intercept_0 = np.dot( np.dot(mu_0.T, sigma_inv), mu_0 )
    constant = 2*math.log(phi/(1-phi)) - intercept_1 + intercept_0

    # Get Separator
    x = np.linspace(60, 180, 250)
    y = [(constant - coeff[0] * xi) / coeff[1] for xi in x]
    plt.plot(x, y, 'r')
    plt.show()



if type is 1:
    # Get individual covariance matrices for PART D
    print ("Covariance matrix for y(i)=0 is \n", sigma_0)
    print ("Covariance matrix for y(i)=1 is \n", sigma_1)




    # Get Quadratic Separator for PART E
    sigma_0_inv = np.linalg.inv(sigma_0)
    sigma_1_inv = np.linalg.inv(sigma_1)
    sigma_diff = sigma_1_inv - sigma_0_inv
    coeff = 2 * (np.dot(mu_0.T, sigma_0_inv) - np.dot(mu_1.T, sigma_1_inv))
    intercept_1 = np.dot( np.dot(mu_1.T, sigma_1_inv), mu_1 )
    intercept_0 = np.dot( np.dot(mu_0.T, sigma_0_inv), mu_0 )
    constant = 2*math.log( (phi*math.sqrt(np.linalg.det(sigma_0))) / ((1-phi)*math.sqrt(np.linalg.det(sigma_1))) ) + intercept_1 - intercept_0

    def z_func(x0, x1):
        quadratic, linear = 0, 0
        quadratic += x0*x0*sigma_diff[0][0]
        quadratic += x0*x1*sigma_diff[0][1]
        quadratic += x1*x0*sigma_diff[1][0]
        quadratic += x1*x1*sigma_diff[1][1]
        linear += coeff[0]*x0
        linear += coeff[1]*x1
        return (quadratic + linear + constant)

    plt.scatter(x_i_0[:, [0]], x_i_0[:, [1]], marker='+')
    plt.scatter(x_i_1[:, [0]], x_i_1[:, [1]], marker='*')

    # Plot Separator
    x0 = np.arange(50, 200, 5)
    x1 = np.arange(250, 550, 5)
    x0, x1 = np.meshgrid(x0, x1)
    z = z_func(x0, x1)
    plt.contour(x0, x1, z, [0])
    plt.show()
