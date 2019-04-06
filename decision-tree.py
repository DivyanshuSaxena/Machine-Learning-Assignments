#!/usr/bin/env python
# coding: utf-8

# In[1]:


import random
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


# In[2]:


def mediate_data(array, median):
    # median = np.median(array)
    preprocess = lambda x : 0 if x < median else 1
    return np.array([preprocess(xi) for xi in array])

def get_values(array):
    unique_values = np.unique(array)
    if 7 in unique_values:
        unique_values = np.array(range(-2, 10))
    return unique_values

def calculate_entropy(array):
    entropy = 0
    for x_tuple in array:
        x_samples = sum(x_tuple)
        x_entropy = 0
        for y_samples in x_tuple:
            x_entropy += (-y_samples*math.log(y_samples/x_samples))
        entropy += x_samples*x_entropy
    return entropy

class Node:
    children = []
    attribute = -1
    sample_indexes = []
    sample_class = 0
    is_leaf = 1
    
    def __init__(self, index_list):
        self.sample_indexes = index_list
        self.children = []
        self.attribute = -1
        self.sample_class = 0
        self.is_leaf = 1
    
root = None


# In[3]:


# Preprocess Data
train_raw = np.genfromtxt('./ass3_data/credit-cards.train.csv', delimiter=',', skip_header=2)
test_raw = np.genfromtxt('./ass3_data/credit-cards.test.csv', delimiter=',', skip_header=2)
val_raw = np.genfromtxt('./ass3_data/credit-cards.val.csv', delimiter=',', skip_header=2)

train_continuous = train_raw[:, [1,2,5,12,13,14,15,16,17,18,19,20,21,22,23]]
test_continuous = test_raw[:, [1,2,5,12,13,14,15,16,17,18,19,20,21,22,23]]
val_continuous = val_raw[:, [1,2,5,12,13,14,15,16,17,18,19,20,21,22,23]]
train_category = train_raw[:, [3,4,6,7,8,9,10,11]]
test_category = test_raw[:, [3,4,6,7,8,9,10,11]]
val_category = val_raw[:, [3,4,6,7,8,9,10,11]]

train_output = train_raw[:, -1]
test_output = test_raw[:, -1]
val_output = val_raw[:, -1]

medians = np.apply_along_axis(np.median, 0, train_continuous)
train_cont_data = np.zeros(len(train_raw))
test_cont_data = np.zeros(len(test_raw))
val_cont_data = np.zeros(len(val_raw))

for i in range(len(train_continuous.T)):
    column = train_continuous.T[i]
    train_cont_data = np.column_stack((train_cont_data, mediate_data(column, medians[i])))

for i in range(len(test_continuous.T)):
    column = test_continuous.T[i]
    test_cont_data = np.column_stack((test_cont_data, mediate_data(column, medians[i])))

for i in range(len(val_continuous.T)):
    column = val_continuous.T[i]
    val_cont_data = np.column_stack((val_cont_data, mediate_data(column, medians[i])))
    
train_cont_data = np.delete(train_cont_data, 0, 1)
test_cont_data = np.delete(test_cont_data, 0, 1)
val_cont_data = np.delete(val_cont_data, 0, 1)

train_data = np.concatenate((train_category, train_cont_data), axis=1)
test_data = np.concatenate((test_category, test_cont_data), axis=1)
val_data = np.concatenate((val_category, val_cont_data), axis=1)

attribute_values = {i: get_values(train_data.T[i]) for i in range(len(train_data[0]))}

num_attributes = len(train_data[0])
print (attribute_values)


# In[4]:


# Part A
m = len(train_data)
# m = 5000
# root = Node(list(range(m)))
epsilon = 0.05
path_attr = []
max_depth = 24
num_nodes = [0] * max_depth

def grow_tree(curr, depth):
    # print ("Starting for node with indexes: {0}".format(curr.sample_indexes)) # Debug
    count_y = []
    for X in attribute_values:
        values = attribute_values[X]
        temp_y = [] 
        for i in range(len(values)):
            temp_y.append([epsilon, epsilon])
        count_y.append(temp_y)

    # Compute y values
    samples0 = samples1 = 0
    for index in curr.sample_indexes:
        y = int(train_output[index])
        if y == 0:
            samples0 += 1
        else:
            samples1 += 1
        for feature_index in range(num_attributes):
            attribute = int(train_data[index][feature_index])
            if feature_index in range(2,8):
                attribute += 2
            count_y[feature_index][attribute][y] += 1
    
    # print ("For the current node, samples0: {0} and samples1: {1}".format(samples0, samples1))
    curr.sample_class = 1 if samples1 > samples0 else 0
    # print ("Giving class {0} for current node".format(curr.sample_class)) # Debug
    if samples0 < 1 or samples1 < 1:
        return 0

    num_nodes[depth] += 1
    
    # Compute entropies of each attribute and choose the best attribute
    best_attribute = -1
    best_entropy = 0
    found_attribute = False
    for X in attribute_values:
        entropy = calculate_entropy(count_y[X])
        if (entropy < best_entropy or best_attribute == -1) and X not in path_attr:
            found_attribute = True
            best_attribute = X
            best_entropy = entropy
            
    if not found_attribute:
        # print ("Exiting because no suitable attribute found")
        return 0

    # print ("Split at attribute {0}".format(best_attribute, count_y[best_attribute])) # Debug
    curr.attribute = best_attribute
    path_attr.append(best_attribute)

    # Make children and append them to the children array
    children_indexes = []
    for i in attribute_values[best_attribute]:
        children_indexes.append([])
    for index in curr.sample_indexes:
        attribute = int(train_data[index][best_attribute])
        if best_attribute in range(2,8):
            attribute += 2
        children_indexes[attribute].append(index)
    for x in attribute_values[best_attribute]:
        if best_attribute in range(2,8):
            x += 2
        child = Node(children_indexes[int(x)])
        curr.children.append(child)
        
    # Recursively grow tree on the children nodes
    for child in curr.children:
        if len(child.sample_indexes) > 0:
            curr.is_leaf = 0
            grow_tree(child, depth+1)
            
    path_attr.remove(best_attribute)
    # print (path_attr)
    return 0


# In[5]:


root = Node(list(range(m)))
grow_tree(root, 0)


# In[6]:


for i in range(1, max_depth):
    num_nodes[i] = num_nodes[i] + num_nodes[i-1]


# In[7]:


# Get Accuracy functions
def get_classification(sample, curr, max_depth, depth):
    # Go to the child based on current best attribute
    if depth == max_depth:
        return curr.sample_class
    # print ("Starting for node with attribute: {0}, class: {1}, children: {3} and values: {2}"
    #       .format(curr.attribute, curr.sample_class, attribute_values[curr.attribute], len(curr.children))) # Debug
    if curr.is_leaf == 1:
        return curr.sample_class
    sample_attribute = sample[curr.attribute]
    if curr.attribute in range(2,8):
        sample_attribute += 2
    return get_classification(sample, curr.children[int(sample_attribute)], max_depth, depth+1)
    
def get_accuracy(dataset, dataset_output, max_depth):
    accuracy = 0
    for index in range(len(dataset)):
        sample = dataset[index]
        model_class = get_classification(sample, root, max_depth, 0)
        if model_class == dataset_output[index]:
            accuracy += 1
    # print (accuracy/len(dataset)*100)
    return (accuracy/len(dataset)*100)


# In[8]:


# Get Training Accuracy
train_accuracy = []
test_accuracy = []
val_accuracy = []

for depth in range(1, max_depth+1):
    train_accuracy.append(get_accuracy(train_data[0:m], train_output[0:m], depth))
    test_accuracy.append(get_accuracy(test_data, test_output, depth))
    val_accuracy.append(get_accuracy(val_data, val_output, depth))
    
# print (train_accuracy, test_accuracy, val_accuracy)

# Plot graph for PART A
plt.plot(num_nodes, train_accuracy, 'b.-', label='Train Accuracy')
plt.plot(num_nodes, test_accuracy, 'r.-', label='Test Accuracy')
plt.plot(num_nodes, val_accuracy, 'g.-', label='Validation Accuracy')
plt.xlabel('Number of Nodes')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


# In[9]:


# Use sklearn for PART D
tree = DecisionTreeClassifier(criterion='entropy')
tree.fit(train_data, train_output)

min_split_tree = DecisionTreeClassifier(criterion='entropy', min_samples_split=800)
min_split_tree.fit(train_raw[:, list(range(1, len(train_raw[0])-1))], train_output)

min_leaf_tree = DecisionTreeClassifier(criterion='entropy', min_samples_leaf=85)
min_leaf_tree.fit(train_raw[:, list(range(1, len(train_raw[0])-1))], train_output)

max_depth_tree = DecisionTreeClassifier(criterion='entropy', max_depth=3)
max_depth_tree.fit(train_raw[:, list(range(1, len(train_raw[0])-1))], train_output)

# print (tree.score(val_raw[:, list(range(1, len(val_raw[0])-1))], val_output))
# print (min_split_tree.score(val_raw[:, list(range(1, len(val_raw[0])-1))], val_output))
# print (min_leaf_tree.score(val_raw[:, list(range(1, len(val_raw[0])-1))], val_output))
# print (max_depth_tree.score(val_raw[:, list(range(1, len(val_raw[0])-1))], val_output))

print ("Train Set Accuracy: {0}".format(min_leaf_tree.score(train_raw[:, list(range(1, len(train_raw[0])-1))], train_output)))
print ("Test Set Accuracy: {0}".format(min_leaf_tree.score(test_raw[:, list(range(1, len(test_raw[0])-1))], test_output)))
print ("Validation Set Accuracy: {0}".format(min_leaf_tree.score(val_raw[:, list(range(1, len(val_raw[0])-1))], val_output)))


# In[10]:


# Get one hot encodings of datasets
concat_category = np.concatenate((np.concatenate((train_category, test_category), axis=0), val_category), axis=0)
concat_df = pd.DataFrame(concat_category)
concat_onehot = pd.get_dummies(concat_df, columns=concat_df.columns)

train_onehot_cat = concat_onehot[:len(train_category)]
test_onehot_cat = concat_onehot[len(train_category):len(train_category)+len(test_category)]
val_onehot_cat = concat_onehot[len(train_category)+len(test_category):]

train_onehot_data = np.concatenate((train_onehot_cat, train_continuous), axis=1)
test_onehot_data = np.concatenate((test_onehot_cat, test_continuous), axis=1)
val_onehot_data = np.concatenate((val_onehot_cat, val_continuous), axis=1)


# In[11]:


# Get accuracies for PART E
onehot_tree = DecisionTreeClassifier(criterion='entropy', min_samples_leaf=85)
onehot_tree.fit(train_onehot_data, train_output)
print ("Train Set Accuracy: {0}".format(onehot_tree.score(train_onehot_data, train_output)))
print ("Test Set Accuracy: {0}".format(onehot_tree.score(test_onehot_data, test_output)))
print ("Validation Set Accuracy: {0}".format(onehot_tree.score(val_onehot_data, val_output)))


# In[12]:


# Get accuracies for PART F
forest = RandomForestClassifier(n_estimators=400, criterion='entropy', max_features=2, bootstrap=True, random_state=5)
forest.fit(train_onehot_data, train_output)
print ("Train Set Accuracy: {0}".format(forest.score(train_onehot_data, train_output)))
print ("Test Set Accuracy: {0}".format(forest.score(test_onehot_data, test_output)))
print ("Validation Set Accuracy: {0}".format(forest.score(val_onehot_data, val_output)))


# In[ ]:




