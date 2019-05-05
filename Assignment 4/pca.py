#!/usr/bin/env python
# coding: utf-8

# In[1]:


import random
import math
import os
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.decomposition import IncrementalPCA


# In[2]:


def load_image(infilename):
    img = Image.open(infilename).convert('L')
    data = np.array(img)
    return data

def view_image(oned_arr):
    image = np.reshape(oned_arr, (210, 160), order='F')
    img = Image.fromarray(image, 'L')
    plt.imshow(img)
    plt.show()


# In[8]:


total_episodes = 0
total_frames = 0
num_frames50 = 0
make_pickle = int(sys.argv[1].rstrip())


# In[9]:


rootdir = "./train_dataset/"
tree = sorted(list(os.walk(rootdir)))

if make_pickle == 1:
    pca_images = []
    for root, sub_folders, files in tree[0:51]:
        print (root)
        if len(files) != 0:
            for file in sorted(files)[0:-1]:
                num_frames50 += 1
                image_file = os.path.join(root, file)
                image_arr = load_image(image_file)
                # img = Image.fromarray(image_arr, 'L')
                # plt.imshow(img)
                # plt.show()
                flattened_image = image_arr.flatten(order='F')
                pca_images.append(flattened_image)
    # all_images = np.array(all_images)
    # print (all_images.shape)
    # pickle_file = './grayscale_pickle'
    # np.save(pickle_file, all_images)
    
    pca_images = np.array(pca_images)
    print ("PCA images done: {0}".format(pca_images.shape))
    ipca = IncrementalPCA(n_components=50, batch_size=200)
    ipca.fit(pca_images)
    with open('pca_model.pickle', 'wb') as handle:
        pickle.dump(ipca, handle, protocol=pickle.HIGHEST_PROTOCOL)


# In[7]:


if not make_pickle == 1:
    with open('./pca_model.pickle', 'rb') as handle:
        model = pickle.load(handle)

    transformed_images = []
    for root, sub_folders, files in tree:
        print (root)
        episode_images = []
        if len(files) != 0:
            for file in sorted(files)[0:-1]:
                total_frames += 1
                image_file = os.path.join(root, file)
                image_arr = load_image(image_file)
                flattened_image = image_arr.flatten(order='F')
                episode_images.append(flattened_image)
            transformed_episode = model.transform(episode_images)
            transformed_images.append(transformed_episode)
            transform_file = './transforms/{0}'.format(root[16:])
            np.save(transform_file, transformed_episode)

            check_transform = np.load("{0}.npy".format(transform_file))

    transformed_images = np.array(transformed_images)
    np.save('./transformed', transformed_images)

# In[ ]:




