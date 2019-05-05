#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pickle
import random
import math
import numpy as np
from PIL import Image
from sklearn.svm import SVC
from sklearn.metrics import f1_score


# In[49]:


def load_image(infilename):
    img = Image.open(infilename).convert('L')
    data = np.array(img)
    return data

# Get Mean and Standard Deviation of the train data
train_data = np.load('./dataset/train.data.npy')
train_output = np.load('./dataset/train.output.npy')
mean = np.mean(train_data)
std = np.std(train_data)


# In[20]:


# Load PCA Model for transforming Images
with open('./models/pca_model.pickle', 'rb') as handle:
    pca_model = pickle.load(handle)
test = True


# In[3]:


if test:
    # Load Test Data
    rootdir = "./test_dataset/"
    tree = sorted(list(os.walk(rootdir)))
else:
    # Load Validation Data
    rootdir = "./validation_dataset/"
    tree = sorted(list(os.walk(rootdir)))


# In[13]:


test_data = []
for root, sub_folders, files in tree[1:]:
    print (root)
    sample = []
    if len(files) != 0:
        for file in sorted(files):
            image_file = os.path.join(root, file)
            image_arr = load_image(image_file)
            flattened_image = image_arr.flatten(order='F')
            transformed_image = pca_model.transform(np.array([flattened_image]))
            sample.append(transformed_image)
        sample = np.array(sample).flatten()
        test_data.append(sample)

if test:
    test_data = np.array(test_data)
    normalized_test = test_data / std
    np.save('./transformed_test', normalized_test)
    print (normalized_test.shape)
else:
    val_data = np.array(test_data)
    normalized_val = val_data / std
    np.save('./transformed_val', normalized_val)
    print (normalized_val.shape)


# In[21]:


# Load the transformed dataset
if test:
    normalized_test = np.load('./transformed_test.npy')
else:
    normalized_val = np.load('./transformed_val.npy')
    
# Load the Model for prediction
with open('./models/svm_linear_lowstd.pickle', 'rb') as handle:
    svm_model = pickle.load(handle)


# In[22]:


if not test:
    reward_file = './validation_dataset/rewards.csv'
    rewards = np.genfromtxt(reward_file, delimiter=',')
    rewards = rewards[:, -1]
    print (rewards.shape)


# In[23]:


if test:
    test_predictions = svm_model.predict(normalized_test)
    indexes = np.arange(len(test_predictions))
    submission = np.column_stack((indexes, test_predictions))
    np.savetxt("./submission.csv", submission.astype(int), delimiter=",", header="id,Prediction", comments='', fmt='%i')
else:
    val_predictions = svm_model.predict(normalized_val)
    f1 = f1_score(rewards, val_predictions)
    print ("F1 Score on Validation Set: {0}".format(f1))


# In[ ]:




