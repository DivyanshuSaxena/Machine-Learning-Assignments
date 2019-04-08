#!/usr/bin/env python
# coding: utf-8

import sys
import math
import copy
import random
import time
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

get_input = sys.argv[1]

if get_input == 1:
    train_file = sys.argv[2]
    test_file = sys.argv[3]
    onehot_train = sys.argv[4]
    onehot_test = sys.argv[5]

    # Get Data
    train_raw = np.genfromtxt(train_file, delimiter=',')
    test_raw = np.genfromtxt(test_file, delimiter=',')

    # One Hot encoding for PART A
    train_df = pd.DataFrame(np.delete(train_raw, -1, 1))
    test_df = pd.DataFrame(np.delete(test_raw, -1, 1))
    train_outraw = train_raw[:, -1]
    test_outraw = test_raw[:, -1]

    train_data = pd.get_dummies(train_df, columns=train_df.columns).values
    test_data = pd.get_dummies(test_df, columns=train_df.columns).values
    train_output = pd.get_dummies(train_outraw).values
    test_output = pd.get_dummies(test_outraw).values

    np.savetxt(onehot_train, train_data, delimiter=",")
    np.savetxt(onehot_test, test_data, delimiter=",")

else:
    config_file = sys.argv[2]
    onehot_train = sys.argv[3]
    onehot_test = sys.argv[4]

    # Get Data
    train_data = np.genfromtxt(onehot_train, delimiter=',')
    test_data = np.genfromtxt(onehot_test, delimiter=',')

    f = open(config_file, "r")
    number_inputs = int(f.readline())
    number_outputs = int(f.readline())
    batch_size = int(f.readline())
    number_layers = int(f.readline())
    hidden_layers = int(f.readline())
    function = f.readline()
    learning_rate = f.readline()

    num_features = len(train_data[0])
    adaptive = (learning_rate == "variable")
    single = (number_layers == 1)
    double = (number_layers == 2)
    relu = (function == "relu")


    def sigmoid(x):
        try:
            val = 1/(1 + math.exp(-x))
        except OverflowError:
            val = 0
        return val

    np.set_printoptions(suppress=True)


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
            if not relu:
                min_epochs = 100
            else:
                min_epochs = 200
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
                    
                    xjk_all = []                                                    
                    # Forward Propagation
                    xjk = copy.deepcopy(batch)
                    for l_index in range(len(self.layers)):
                        layer_weights = self.layers[l_index]
                        xjk_complete = np.insert(xjk, len(xjk[0]), 1, axis=1)
                        xjk_all.append(xjk_complete)
                        o_j = np.matmul(np.delete(layer_weights, -1, 1), np.array(xjk).T).T + np.array([layer_weights[:, -1]])
                        if relu and l_index != len(self.layers)-1:
                            xjk = np.maximum(0, o_j)
                        else:
                            xjk = np.vectorize(sigmoid)(o_j)
                    output = xjk

                    if debug_nn:
                        print ("Forward Feed complete.")

                    # Calculate error
                    loss_func += np.sum((batch_target-output)*(batch_target-output))
                    
                    # Back Propagation
                    updated_weights = copy.deepcopy(self.layers)
                    layer_weights = self.layers[-1]
                    xjk = xjk_all[-1]
                    # print ("Dimension of xjk: {0}".format(xjk.shape))
                    del_netj = (batch_target-output)*output*(1-output)
                    # print ("Dimension of del_netj: {0}".format(del_netj.shape))
                    del_j = np.matmul(del_netj.T, xjk)              
                    updated_weights[-1] = np.add(layer_weights, self.learning_rate*del_j)
                        
                    # Hidden Layers
                    for layer in reversed(range(len(self.layers[:-1]))):  
                        layer_weights = self.layers[layer]
                        xjk = xjk_all[layer]
                        # print ("Dimension of xjk: {0}".format(xjk.shape))
                        o_j = np.delete(xjk_all[layer+1], -1, 1)
                        del_lj = np.matmul(del_netj, np.delete(self.layers[layer+1], -1, 1))
                        # print ("Dimension of del_lj: {0}".format(del_lj.shape))
                        if relu:
                            del_netj = np.multiply(del_lj, np.maximum(0, o_j)/np.maximum(0.00000001, o_j))
                        else:
                            del_netj = np.multiply(del_lj, (o_j*(1-o_j)))
                        # print ("Dimension of del_netj: {0}".format(del_netj.shape))
                        del_j = np.matmul(del_netj.T, xjk)
                        # print ("Dimension of del_j: {0}".format(del_j.shape))                    
                        updated_weights[layer] = np.add(layer_weights, self.learning_rate*del_j)
                    self.layers = updated_weights

                    if debug_nn:
                        print ("Updated weights: {0}".format(self.layers))
                        print ("-----------------------------------------")
                    batch = []
                    batch_target = []
                
                if abs(prev_loss-loss_func)/len(dataset_shuffle) < 0.00001:
                    if not adaptive:
                        if epoch > min_epochs:
                            break
                    else:
                        num_useless_epochs += 1
                        if num_useless_epochs == 2:
                            self.learning_rate = self.learning_rate/5
                        if self.learning_rate < 0.001:
                            break
                prev_loss = loss_func
                if debug_nn:
                    print ("Loss: {0}   Iteration: {1}".format(loss_func, epoch))
            if debug_nn:
                print ("Final Weights: {0}".format(self.layers))
            print ("Num Epochs: {0}".format(epoch+1))
            
        def predict(self, batch):
            # Forward Propagation
            xjk = copy.deepcopy(batch)
            for l_index in range(len(self.layers)):
                layer_weights = self.layers[l_index]
                net_j = np.matmul(np.delete(layer_weights, -1, 1), np.array(xjk).T).T + np.array([layer_weights[:, -1]])
                if relu and l_index != len(self.layers)-1:
                    xjk = np.maximum(0, net_j)
                else:
                    xjk = np.vectorize(sigmoid)(net_j)
            output = xjk
            return output
        
        def get_accuracy(self, dataset, target):
            accuracy = 0
            confusion_matrix = np.zeros(shape=(self.num_outputs, self.num_outputs))
            output = self.predict(dataset)
            prediction = np.argmax(output, axis=1)
            for s_index in range(len(dataset)):
                pred = prediction[s_index]
                if debug_nn:
                    print ("Predicted: {0} Desired: {1}".format(pred, target[s_index]))
                confusion_matrix[int(target[s_index])][pred] += 1
                if pred == target[s_index]:
                    accuracy += 1
            print ("Accuracy: {0}".format(accuracy/len(dataset)))
            print (confusion_matrix)
            return accuracy/len(dataset)*100, confusion_matrix


    def get_plots(hidden_units, m, test_m, name, color):
        train_accuracy = []
        test_accuracy  = []
        train_time     = []
        train_cm       = []
        test_cm        = []
        for num_units in hidden_units:
            nn = Neural_Net(100, num_features, num_units, 10)
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
        
        # Save figures
        xi = [i for i in range(0, len(hidden_units))]
        plt.figure()
        plt.plot(xi, train_time, 'b.-')
        plt.xticks(xi, hidden_units)
        plt.xlabel("Number of Perceptrons in Hidden Layer")
        plt.ylabel("Train Time")
        plt.ylim(30,300)
        plt.savefig('./{0}_train_time.png'.format(name))
        
        xi = [i for i in range(0, len(hidden_units))]
        plt.figure()
        plt.plot(xi, train_accuracy, 'r.-', label='Train Accuracy')
        plt.plot(xi, test_accuracy, 'g.-', label='Test Accuracy')
        plt.xticks(xi, hidden_units)
        plt.xlabel("Number of Perceptrons in Hidden Layer")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.ylim(50,100)
        plt.savefig('./{0}_accuracy.png'.format(name))

        # Plot Confusion Matrices
        units = [5, 10, 15, 20, 25]
        for cm_index in range(len(test_cm)):
            cm = test_cm[cm_index]
            plt.figure()
            ax = sns.heatmap(cm, cmap=color)
            fig = ax.get_figure()
            fig.savefig('./{0}{1}.png'.format(name, units[cm_index]))

        print (train_accuracy, test_accuracy)
        return train_accuracy, test_accuracy, train_time, train_cm, test_cm


    # m = 100
    # test_m = 100
    m = len(train_data)
    test_m = len(test_data)


    # Get Plots for PART C, D, E and F
    hidden_units_single = [[5],[10],[15],[20],[25]]
    hidden_units_double = [[5,5],[10,10],[15,15],[20,20],[25,25]]

    if single:
        get_plots(hidden_units_single, m, test_m, 'c', "PuBuGn")
    elif double:
        get_plots(hidden_units_double, m, test_m, 'd', "YlOrBr")
    elif relu:
    #     get_plots(hidden_units_single, m, test_m, 'fc', "PuBuGn")
        get_plots(hidden_units_double, m, test_m, 'fd', "YlOrBr")
    else:
        get_plots(hidden_units_single, m, test_m, 'ec', "PuBuGn")
        get_plots(hidden_units_double, m, test_m, 'ed', "YlOrBr")

