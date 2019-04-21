#!/usr/bin/env python
# coding: utf-8

# In[20]:


import os
import pickle
import random
import math
import numpy as np
from PIL import Image
from sklearn.svm import SVC
from sklearn.metrics import f1_score


# In[8]:


def load_image(infilename):
    img = Image.open(infilename).convert('L')
    data = np.array(img)
    return data

# Get Mean and Standard Deviation of the train data
train_data = np.load('./dataset/train.data.npy')
train_output = np.load('./dataset/train.output.npy')
mean = np.mean(train_data, axis=0)
std = np.std(train_data, axis=0)


# In[19]:


# Load PCA Model for transforming Images
with open('./pca_model.pickle', 'rb') as handle:
    pca_model = pickle.load(handle)

# Load Test Data
rootdir = "./test_dataset/"
tree = sorted(list(os.walk(rootdir)))

test_data = []
for root, sub_folders, files in tree:
    print (root)
    sample = []
    if len(files) != 0:
        for file in sorted(files):
            image_file = os.path.join(root, file)
            image_arr = load_image(image_file)
            flattened_image = image_arr.flatten(order='F')
            sample.append(flattened_image)
        sample = pca_model.transform(sample)
        sample = np.array(sample).flatten()
        test_data.append(sample)

test_data = np.array(test_data)
normalized_test = (test_data - mean) / std
np.save('./transformed_test', normalized_test)
print (normalized_test.shape)


# In[32]:


# Load the Model for prediction
with open('./svm_model.pickle', 'rb') as handle:
    svm_model = pickle.load(handle)

test_predictions = svm_model.predict(normalized_test)
indexes = np.arange(len(test_predictions))


# In[36]:


submission = np.column_stack((indexes, test_predictions))
np.savetxt("./submissions.csv", submission, delimiter=",", header="id, Prediction", comments='')

