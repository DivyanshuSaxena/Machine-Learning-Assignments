#!/usr/bin/env python
# coding: utf-8

# In[51]:


import os
import sys
import math
import time
import random
import pickle
import numpy as np
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score


generate = int(sys.argv[1].rstrip())
svm_type = int(sys.argv[2].rstrip())
linear = (svm_type == 1)

if generate == 1:
    print ("Generating dataset")
    transformed_images = np.load('transformed.npy')


    rootdir = "./train_dataset/"
    tree = sorted(list(os.walk(rootdir)))

    rewards = []
    for root, sub_folders, files in tree:
            print (root)
            if len(files) != 0:
                reward_file = os.path.join(root, 'rew.csv')
                episode_reward = np.genfromtxt(reward_file)
                episode_reward = np.append(episode_reward, 0.0)
                rewards.append(episode_reward)
    rewards = np.array(rewards)



    # Construct the training, test and validation data
    train_data = []
    train_output = []
    test_data = []
    test_output = []
    validation_data = []
    validation_output = []

    iteration_index = list(range(len(transformed_images)))
    random.shuffle(iteration_index)

    num0 = 0
    num1 = 0

    for e_index in iteration_index:
        print (e_index)
        episode = transformed_images[e_index]
        episode_rewards = rewards[e_index]
        for f_index in range(6, len(episode)):
            reward = episode_rewards[f_index]
            if reward == 0:
                # Generate only one example for a sample with reward zero
                num0 += 1
                frame_indexes = np.sort(np.random.choice(6, 4, replace=False))
                frame = []
                for index in frame_indexes:
                    frame.append(episode[f_index-6+index])
                frame.append(episode[f_index])
                frame = np.array(frame).flatten()
                prob = np.random.uniform()
                if prob < 0.2:
                    validation_data.append(frame)
                    validation_output.append(reward)
                elif prob < 0.5:
                    test_data.append(frame)
                    test_output.append(reward)
                else:
                    train_data.append(frame)
                    train_output.append(reward)
            else:
                # Generate all possible examples
                num1 += 1
                indexes = np.arange(6)
                combinations = list(itertools.combinations(indexes, 4))
                sample_reward = np.ones(len(combinations))
                sample_data = []
                for frame_indexes in combinations:
                    frame = []
                    for index in frame_indexes:
                        frame.append(episode[f_index-6+index])
                    frame.append(episode[f_index])
                    frame = np.array(frame).flatten()
                    sample_data.append(frame)
                prob = np.random.uniform()
                if prob < 0.2:
                    validation_data.extend(sample_data)
                    validation_output.extend(sample_reward)
                elif prob < 0.5:
                    test_data.extend(sample_data)
                    test_output.extend(sample_reward)
                else:
                    train_data.extend(sample_data)
                    train_output.extend(sample_reward)

    train_data = np.array(train_data)
    train_output = np.array(train_output)
    test_data = np.array(test_data)
    test_output = np.array(test_output)
    validation_data = np.array(validation_data)
    validation_output = np.array(validation_output)
    print (train_data.shape)
    print (test_data.shape)
    print (validation_data.shape)
    print (num0, num1)


    # # Save data files
    np.save('./dataset/train.data', train_data)
    np.save('./dataset/train.output', train_output)
    np.save('./dataset/test.data', test_data)
    np.save('./dataset/test.output', test_output)
    np.save('./dataset/val.data', validation_data)
    np.save('./dataset/val.output', validation_output)

else:
    # Load Data files
    train_data = np.load('./dataset/train.data.npy')
    train_output = np.load('./dataset/train.output.npy')
    test_data = np.load('./dataset/test.data.npy')
    test_output = np.load('./dataset/test.output.npy')
    validation_data = np.load('./dataset/val.data.npy')
    validation_output = np.load('./dataset/val.output.npy')


    # Normalize the dataset
    # std = train_data.std()
    std = 250
    print (std)
    normalized_train = train_data / std
    normalized_test = test_data / std
    normalized_val = validation_data / std


    # Fit the model and dump it
    num_examples = 70000
    if linear:
        model = LinearSVC(tol=0.00001*num_examples, max_iter=7500)
    else:
        model = SVC(kernel='rbf', class_weight='balanced', tol=0.00001*num_examples, max_iter=7500)
    start_time = time.time()
    idx = np.random.choice(np.arange(len(normalized_train)), num_examples, replace=False)
    model.fit(normalized_train[idx], train_output[idx])
    end_time = time.time()
    print ("Time taken for training: {0}".format(end_time-start_time))
    if linear:
        with open('./models/svm_linear_lowstd.pickle', 'wb') as handle:
            pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open('./models/svm_gaussian_unnorm.pickle', 'wb') as handle:
            pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)


    # Evaluate over train set
    eval_examples = 20000
    train_idx = np.random.choice(np.arange(len(normalized_train)), eval_examples, replace=False)
    acc = model.score(normalized_train[train_idx], train_output[train_idx])
    print ("Accuracy on Train Set: {0}".format(acc))
    train_predictions = model.predict(normalized_train[train_idx])
    f1 = f1_score(train_output[train_idx], train_predictions)
    print ("F1 Score on Train Set: {0}".format(f1))

    # Evaluate over validation set
    val_idx = np.random.choice(np.arange(len(normalized_val)), eval_examples, replace=False)
    acc = model.score(normalized_val[val_idx], validation_output[val_idx])
    print ("Accuracy on Validation Set: {0}".format(acc))
    val_predictions = model.predict(normalized_val[val_idx])
    f1 = f1_score(validation_output[val_idx], val_predictions)
    print ("F1 Score on Validation Set: {0}".format(f1))

    # Score over test set
    test_idx = np.random.choice(np.arange(len(normalized_test)), eval_examples, replace=False)
    acc = model.score(normalized_test[test_idx], test_output[test_idx])
    print ("Accuracy on Test Set: {0}".format(acc))
    test_predictions = model.predict(normalized_test[test_idx])
    f1 = f1_score(test_output[test_idx], test_predictions)
    print ("F1 Score on Test Set: {0}".format(f1))
