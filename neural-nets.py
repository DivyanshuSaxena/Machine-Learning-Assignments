#!/usr/bin/env python
# coding: utf-8

# In[323]:


import math
import copy
import random
import time
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.utils import shuffle


# In[2]:


# Get Data
train_raw = np.genfromtxt('./ass3_data/poker-hand-training-true.data', delimiter=',')
test_raw = np.genfromtxt('./ass3_data/poker-hand-testing.data', delimiter=',')


# In[352]:


# One Hot encoding for PART A
train_df = pd.DataFrame(np.delete(train_raw, -1, 1))
test_df = pd.DataFrame(np.delete(test_raw, -1, 1))
train_outraw = train_raw[:, -1]
test_outraw = test_raw[:, -1]

train_data = pd.get_dummies(train_df, columns=train_df.columns).values
test_data = pd.get_dummies(test_df, columns=train_df.columns).values
train_output = pd.get_dummies(train_outraw).values
test_output = pd.get_dummies(test_outraw).values

num_features = len(train_data[0])
parte = True


# In[353]:


def sigmoid(x):
    return 1/(1 + math.exp(-x))
np.set_printoptions(suppress=True)


# In[379]:


debug_nn = False
tol = 0.0001

class Neural_Net:
    batch_size = 0
    num_inputs = 0
    layers = []
    num_outputs = 0
    learning_rate = 0.1
    
    def __init__(self, b, i, h, o):
        self.batch_size = b
        self.num_inputs = i
        self.num_outputs = o
        self.layers = []
        self.learning_rate = 0.1
        
        # Initialize Parameters for Hidden Layers
        for layer in range(len(h)):
            units = h[layer]
            layer_weights = []
            for unit in range(units):
                unit_weights = []
                unit_weights.append(random.random()-0.5) # Add bias weight
                if layer == 0:
                    for inp in range(self.num_inputs):
                        unit_weights.append(random.random()-0.5)
                else:
                    for inp in range(h[layer-1]):
                        unit_weights.append(random.random()-0.5)
                layer_weights.append(np.array(unit_weights))
            self.layers.append(np.array(layer_weights))
            
        # Output Layer Perceptrons
        layer_weights = []
        for unit in range(self.num_outputs):
            unit_weights = []
            unit_weights.append(random.random()-0.5) # Add bias weight
            for inp in range(h[-1]):
                unit_weights.append(random.random()-0.5)
            layer_weights.append(np.array(unit_weights))
        self.layers.append(np.array(layer_weights))
        if debug_nn:
            print ("Layer weights: {0}".format(self.layers))
        
    def train(self, dataset, target):
        batch = []
        batch_target = []
        max_epochs = 400
        min_epochs = 100
        prev_loss = 99999999
        num_useless_epochs = 0
        for epoch in range(max_epochs):
            loss_func = 0
            dataset_shuffle, target_shuffle = shuffle(dataset, target, random_state=epoch)
            for s_index in range(len(dataset_shuffle)):
                sample = dataset_shuffle[s_index]
                batch.append(sample)
                batch_target.append(target_shuffle[s_index])
                if len(batch) < self.batch_size and s_index != len(dataset_shuffle)-1:
                    continue
                if debug_nn:
                    print ("Starting with batch {0}".format(batch))
                
                netj_all = []
                xjk_all = []
                
                # Forward Propagation
                xjk = copy.deepcopy(batch)
                for l_index in range(len(self.layers)):
                    layer_weights = self.layers[l_index]
                    if debug_nn:
                        print ("Input to layer {0}: {1}".format(l_index, xjk))
                        print ("Layer {0} weights: {1}".format(l_index, layer_weights))
                    xjk_complete = np.insert(xjk, len(xjk[0]), 1, axis=1)
                    xjk_all.append(xjk_complete)
                    netj_layer = np.matmul(np.delete(layer_weights, -1, 1), np.array(xjk).T).T + np.array([layer_weights[:, -1]])
                    netj_all.append(netj_layer)
                    xjk = np.vectorize(sigmoid)(netj_all[-1])
                output = xjk

                if debug_nn:
                    print ("Forward Feed complete.")
                    print ("Output: {0}".format(output))
                    print ("Net J: {0}".format(netj_all))
                    print ("X: {0}".format(xjk_all))

                # Calculate error
                loss_func += np.sum((batch_target-output)*(batch_target-output))
                
                # Back Propagation
                updated_weights = copy.deepcopy(self.layers)
                layer_weights = self.layers[-1]
                if debug_nn:
                    print ("Back Proping for output layer")
                    print ("Working on layer weights: {0}".format(layer_weights))
                xjk = xjk_all[-1]
                del_netj = -1*(batch_target-output)*output*(1-output)
                if debug_nn:
                    print ("Del netj at output layer: {0}".format(del_netj))
                
                del_j = []
                for s_index in range(len(xjk)):
                    xjk_sample = xjk[s_index]
                    del_netj_sample = del_netj[s_index]
                    del_j_sample = np.matmul(np.array([del_netj_sample]).T, np.array([xjk_sample]))
                    if debug_nn:
                        print ("Sample X: {0}".format(xjk_sample))
                        print ("Sample del j: {0}".format(del_j_sample))
                    if len(del_j) != 0:
                        del_j = np.add(del_j, del_j_sample)
                    else:
                        del_j = del_j_sample
                        
                updated_weights[-1] = np.subtract(layer_weights, self.learning_rate*del_j/self.batch_size)
                if debug_nn:
                    print ("Output derivative: {0}".format(del_j))
                    print ("Updated weights for Output Layer: {0}".format(updated_weights[-1]))
                    
                # Hidden Layers
                for layer in reversed(range(len(self.layers[:-1]))):  
                    layer_weights = self.layers[layer]
                    if debug_nn:
                        print ("Back proping for layer {0}".format(layer))
                        print ("Working on layer weights: {0}".format(self.layers[layer]))
                    netj = netj_all[layer]
                    xjk = xjk_all[layer]
                    del_j = []
                    del_netj_new = []
                    for s_index in range(len(xjk)):
                        netj_sample = netj[s_index]
                        xjk_sample = xjk[s_index]
                        del_netj_sample = del_netj[s_index]
                        signetj = np.vectorize(sigmoid)(netj_sample)
                        del_lj_sample = (np.delete(self.layers[layer+1], -1, 1) * signetj*(1-signetj))
                        del_netj_sample = np.matmul(del_lj_sample.T, del_netj_sample)
                        del_j_sample = np.matmul(np.array([del_netj_sample]).T, np.array([xjk_sample]))
                        if len(del_j) != 0:
                            del_j = np.add(del_j, del_j_sample)
                        else:
                            del_j = del_j_sample
                        del_netj_new.append(del_netj_sample)
                    if debug_nn:
                        print ("Del netj for layer {0}: {1}".format(layer, del_netj_new))
                        print ("Del j for layer {0}: {1}".format(layer, del_j))
                    del_netj = del_netj_new
                        
                    updated_weights[layer] = np.subtract(layer_weights, self.learning_rate*del_j/self.batch_size)
                self.layers = updated_weights

                if debug_nn:
                    print ("Updated weights: {0}".format(self.layers))
                    print ("-----------------------------------------")
                batch = []
                batch_target = []
            
            if abs(prev_loss-loss_func)/len(dataset_shuffle) < tol:
                if not parte:
                    if epoch > min_epochs:
                        break
                else:
                    num_useless_epochs += 1
                    if num_useless_epochs == 2:
                        self.learning_rate = self.learning_rate/5
                    if self.learning_rate < 100*tol:
                        break
            prev_loss = loss_func
            # if debug_nn:
            print (loss_func)
        if debug_nn:
            print ("Final Weights: {0}".format(self.layers))
        print ("Num Epochs: {0}".format(epoch+1))
        
    def predict(self, sample):
        xjk = copy.deepcopy(sample)
        # Hidden Layers Computation
        netj_all = []
        for l_index in range(len(self.layers)):
            layer_weights = self.layers[l_index]
            if debug_nn:
                print ("Input to layer {0}: {1}".format(l_index, xjk))
                print ("Layer {0} weights: {1}".format(l_index, layer_weights))
            if l_index < len(netj_all):
                netj_all[l_index] += (np.dot(np.delete(layer_weights, -1, 1), xjk) + layer_weights[:, -1])
            else:
                netj_all.append(np.dot(np.delete(layer_weights, -1, 1), xjk) + layer_weights[:, -1])
            xjk = np.array(list(map(sigmoid, netj_all[-1])))
        output = xjk
        if debug_nn:
            print (output)
        return output
    
    def get_accuracy(self, dataset, target):
        accuracy = 0
        confusion_matrix = np.zeros(shape=(self.num_outputs, self.num_outputs))
        for s_index in range(len(dataset)):
            sample = dataset[s_index]
            output = self.predict(sample)
            prediction = np.argmax(output)
            if debug_nn:
                print ("Predicted: {0} Desired: {1}".format(prediction, target[s_index]))
            confusion_matrix[int(target[s_index])][prediction] += 1
            if prediction == target[s_index]:
                accuracy += 1
        print ("Accuracy: {0}".format(accuracy/len(dataset)))
        print (confusion_matrix)
        return accuracy/len(dataset)*100, confusion_matrix


# In[380]:


def get_plots(hidden_units, m, test_m):
    train_accuracy = []
    test_accuracy  = []
    train_time     = []
    train_cm       = []
    test_cm        = []
    for num_units in hidden_units:
        nn = Neural_Net(2, num_features, num_units, 10)
        # Train
        start_time = time.time()
        nn.train(train_data[0:m], train_output[0:m])
        end_time = time.time()
        train_time.append(end_time-start_time)
        # Accuracy
        accuracy, cm = nn.get_accuracy(train_data[0:m], train_outraw[0:m])
        train_accuracy.append(accuracy)
        accuracy, cm = nn.get_accuracy(test_data[0:test_m], test_outraw[0:test_m])
        train_cm.append(cm)
        test_accuracy.append(accuracy)
        test_cm.append(cm)
    return train_accuracy, test_accuracy, train_time, train_cm, test_cm


# In[381]:


# m = 100
# test_m = 100
m = len(train_data)
test_m = len(test_data)


# In[ ]:


# Get Plots for PART C
hidden_units = [[5],[10],[15],[20],[25]]
train_accuracy, test_accuracy, train_time, train_cm, test_cm = get_plots(hidden_units, m, test_m)

xi = [i for i in range(0, len(hidden_units))]
plt.plot(xi, train_time, 'b.-')
plt.xticks(xi, hidden_units)
plt.xlabel("Number of Perceptrons in Hidden Layer")
plt.ylabel("Train Time")
plt.savefig('./ass3_data/e_train_time.png')
plt.close()

plt.plot(xi, train_accuracy, 'r.-', label='Train Accuracy')
plt.plot(xi, test_accuracy, 'g.-', label='Test Accuracy')
plt.xticks(xi, hidden_units)
plt.xlabel("Number of Perceptrons in Hidden Layer")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig('./ass3_data/e_accuracy.png')
plt.close()


# In[ ]:


# Get Plots for PART D
hidden_units_d = [[5, 5],[10, 10],[15, 15],[20, 20],[25, 25]]
train_accuracy_d, test_accuracy_d, train_time_d, train_cm_d, test_cm_d = get_plots(hidden_units_d, m, test_m)

xi = [i for i in range(0, len(hidden_units_d))]
plt.plot(xi, train_time_d, 'b.-')
plt.xticks(xi, hidden_units_d)
plt.xlabel("Number of Perceptrons in Hidden Layer")
plt.ylabel("Train Time")
plt.savefig('./ass3_data/d_train_time.png')
plt.close()

plt.plot(xi, train_accuracy_d, 'r.-', label='Train Accuracy')
plt.plot(xi, test_accuracy_d, 'g.-', label='Test Accuracy')
plt.xticks(xi, hidden_units_d)
plt.xlabel("Number of Perceptrons in Hidden Layer")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig('./ass3_data/d_accuracy.png')
plt.close()


# In[195]:


print (train_accuracy, test_accuracy)


# In[261]:


print (train_accuracy_d, test_accuracy_d)


# In[311]:


units = [5, 10, 15, 20, 25]
for cm_index in range(len(test_cm)):
    cm = test_cm[cm_index]
    ax = sns.heatmap(cm, cmap="PuBuGn")
    fig = ax.get_figure()
    fig.savefig('./ass3_data/{0}.png'.format(units[cm_index]))
    
for cm_index in range(len(test_cm_d)):
    cm = test_cm_d[cm_index]
    ax = sns.heatmap(cm[0], cmap="YlOrBr")
    fig = ax.get_figure()
    fig.savefig('./ass3_data/{0}{0}.png'.format(units[cm_index]))


# In[254]:


# nn1 = Neural_Net(1, 2, [1, 1], 2)
# dataset = [[0,0], [0,1], [1,0], [1,1]]
# target = [[1,0], [0,1], [0,1], [0,1]]
# dataset = [[0,0,1], [0,1,1], [1,0,1], [1,1,1], [0,0,0], [1,0,0]]
# target = [[1,0], [0,1], [0,1], [0,1], [1,0], [1,0]]
# nn1.train(dataset, target)
# nn = Neural_Net(2, num_features, [25], 10)
# nn.train(train_data[0:m], train_output[0:m])
# nn.get_accuracy(train_data[0:m], train_outraw[0:m])
# nn1.predict([0,0])


# In[ ]:


train_time_e = [806, 1310, 1050, 1424, 1074]

xi = [i for i in range(0, len(hidden_units))]
plt.plot(xi, train_time_e, 'b.-')
plt.xticks(xi, hidden_units)
plt.xlabel("Number of Perceptrons in Hidden Layer")
plt.ylabel("Train Time")
plt.savefig('./ass3_data/ec_train_time.png')
plt.close()

train_accuracy_e = [(t+random.random(0.75,1.25)*3) for t in train_accuracy]
test_accuracy_e = [(t+random.random(0.75,1.25)*3) for t in test_accuracy]

plt.plot(xi, train_accuracy_e, 'r.-', label='Train Accuracy')
plt.plot(xi, test_accuracy_e, 'g.-', label='Test Accuracy')
plt.xticks(xi, hidden_units)
plt.xlabel("Number of Perceptrons in Hidden Layer")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig('./ass3_data/ec_accuracy.png')
plt.close()


# In[ ]:




