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
from sklearn.metrics import f1_score


generate = int(sys.argv[1].rstrip())

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
    mean = np.mean(train_data, axis=0)
    std = np.std(train_data, axis=0)
    normalized_train = (train_data - mean) / std
    normalized_test = (test_data - mean) / std
    normalized_val = (validation_data - mean) / std


    # Fit the model and dump it
    num_examples = 100000
    model_linear = SVC(kernel='linear', class_weight='balanced', tol=0.00001*num_examples, max_iter=7500)
    start_time = time.time()
    model_linear.fit(normalized_train[0:num_examples], train_output[0:num_examples])
    end_time = time.time()
    print ("Time taken for training: {0}".format(end_time-start_time))
    with open('./models/svm_model.pickle', 'wb') as handle:
        pickle.dump(model_linear, handle, protocol=pickle.HIGHEST_PROTOCOL)


    # Evaluate over train set
    eval_examples = 20000
    acc = model_linear.score(normalized_train[num_examples:num_examples+eval_examples], test_output[num_examples:num_examples+eval_examples])
    print ("Accuracy on Train Set: {0}".format(acc))
    train_predictions = model_linear.predict(normalized_train[num_examples:num_examples+eval_examples])
    f1 = f1_score(train_output[num_examples:num_examples+eval_examples], train_predictions)
    print ("F1 Score on Train Set: {0}".format(f1))

    # Evaluate over validation set
    acc = model_linear.score(normalized_val[0:eval_examples], validation_output[0:eval_examples])
    print ("Accuracy on Validation Set: {0}".format(acc))
    val_predictions = model_linear.predict(normalized_val[0:eval_examples])
    f1 = f1_score(validation_output[0:eval_examples], val_predictions)
    print ("F1 Score on Validation Set: {0}".format(f1))

    # Score over test set
    acc = model_linear.score(normalized_test[0:eval_examples], test_output[0:eval_examples])
    print ("Accuracy on Test Set: {0}".format(acc))
    test_predictions = model_linear.predict(normalized_test[0:eval_examples])
    f1 = f1_score(test_output[0:eval_examples], test_predictions)
    print ("F1 Score on Test Set: {0}".format(f1))
