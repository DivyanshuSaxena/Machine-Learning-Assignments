#!/usr/bin/env python
# coding: utf-8

# In[70]:


import json
import nltk
import math
import random
import re
import time
import numpy as np
from nltk.tokenize import RegexpTokenizer, TweetTokenizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk import bigrams

# Download Punkt and Stopwords
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    
# tokenizer = RegexpTokenizer(r'\w+')
tokenizer = TweetTokenizer()
stop_words = list(set(stopwords.words('english')))

# Hyper parameters
profile = True

# Parameters for PART D
use_stop_words = False
use_stemming = False

# Parameters for PART E
use_bigrams = False
if use_bigrams:
    punctuation = ['.', ',', ':', '(', ')', '?', '<', '?', '>', '\'', '\"']
    stop_words.extend(punctuation)
ps = PorterStemmer()


# In[71]:


train_data = []
test_data = []

with open('ass2_data/train.json') as f:
    train_data = json.load(f)
        
with open('ass2_data/test.json') as f:
    test_data = json.load(f)


# In[72]:


vocab = {}
word_index = 0
m = len(train_data)
# m = 100000

def get_vocab():
    global word_index
    vocab_text = open('ass2_data/vocab.txt', 'w')
    for example in train_data:
        # Add single words in vocab
        review = []
        # print (example["text"])
        for word in tokenizer.tokenize(example["text"]):
            word = word.lower()
            word = word.replace('\n', '')
            if use_stop_words and word in stop_words:
                continue
            review.append(word)
            if use_stemming:
                word = ps.stem(word)
            if word not in vocab:
                vocab_text.write(word + "\n")
                vocab[word] = word_index
                word_index += 1
        # Add bigrams in vocab
        if use_bigrams:
            bigrams = list(nltk.bigrams(review))
            # print (bigrams)
            for bigram in bigrams:
                insert_bigram = bigram[0] + "_" + bigram[1] 
                if insert_bigram not in vocab:
                    vocab_text.write(insert_bigram + "\n")
                    vocab[insert_bigram] = word_index
                    word_index += 1
        
    vocab_text.close()
    print (len(vocab.keys()))

try:
    with open('ass2_data/vocab.txt') as f:
        temp_vocab = f.readlines()
        temp_vocab = [x.rstrip() for x in temp_vocab]
        for word in temp_vocab:
            vocab[word] = word_index
            word_index += 1
    print (len(vocab))
except FileNotFoundError:
    if profile:
        get_ipython().run_line_magic('prun', 'get_vocab()')
    else:
        get_vocab()


# In[73]:


# Constants for the algorithm
V = len(vocab.keys())
print (V)

thetas = [1] * (V*5)
phis = [0] * 5
lengths = [V] * 5
counts = [0] * 5

# Apply Naive Bayes algorithm with simplified multinomial assumption 
# over the jth index word of ith document 
def evaluate_thetas():
    for example in train_data[0:m]:
        stars = int(example["stars"])
        counts[stars-1] += 1
        lengths[stars-1] += len(example["text"])
        review = []
        for word in tokenizer.tokenize(example["text"]):
            word = word.lower()
            word = word.replace('\n', '')
            if use_stop_words and word in stop_words:
                continue
            review.append(word)
            if use_stemming:
                word = ps.stem(word)
            k = vocab[word]
            thetas[(stars-1)*V + k] += 1
        if use_bigrams:
            bigrams = list(nltk.bigrams(review))
            for bigram in bigrams:
                k = vocab[bigram[0]+'_'+bigram[1]]
                # print (bigram[0]+'_'+bigram[1], k)
                thetas[(stars-1)*V + k] += 1
            
if profile:
    get_ipython().run_line_magic('prun', 'evaluate_thetas()')
else:
    evaluate_thetas()


# In[74]:


# Make final evaluations for the parameters
log_thetas = [0] * (V*5)
for k in range(V*5):
    star_index = int(k/V)
    log_thetas[k] = math.log(thetas[k]/lengths[star_index])
    
for c in range(5):
    phis[c] = counts[c]/m


# In[76]:


test_m = len(test_data)
# test_m = 500
confusion_matrix = np.zeros(shape=(5,5))

def get_accuracy():
    # Get the accuracy of the trained model over the training data and test data for PART A
    accuracy = 0
    for example in train_data[0:m]:
        max_ll = 0
        max_ll_stars = 0
        for y in range(1,6):
            ll = math.log(phis[y-1])
            review = []
            for word in tokenizer.tokenize(example["text"]):
                word = word.lower()
                word = word.replace('\n', '')
                if use_stop_words and word in stop_words:
                    continue
                review.append(word)
                if use_stemming:
                    word = ps.stem(word)
                try:
                    k = vocab[word]
                    ll += log_thetas[V*(y-1) + k]
                except:
                    ll += math.log(1/V)
            if use_bigrams:
                bigrams = list(nltk.bigrams(review))
                for bigram in bigrams:
                    try:
                        k = vocab[bigram[0]+'_'+bigram[1]]
                        ll += log_thetas[V*(y-1) + k]
                    except:
                        ll += math.log(1/V)
                        
            if (max_ll == 0 or max_ll < ll):
                max_ll = ll
                max_ll_stars = y

        # Calculated the max likelihood stars. Now check accuracy
        # print ("{0} {1}".format(max_ll_stars, example["stars"]))
        if max_ll_stars == example["stars"]:
            accuracy += 1
            
        # Evaluate confusion matrix for PART C
        confusion_matrix[max_ll_stars-1][int(example["stars"])-1] += 1

    print (accuracy/m * 100)
    print (np.matrix(confusion_matrix))
    
if profile:
    get_ipython().run_line_magic('prun', 'get_accuracy()')
else:
    get_accuracy()


# In[69]:


# Calculate F1 Scores for PART F
f1_scores = np.zeros(5)
for i in range(5):
    precision = confusion_matrix[i][i] / confusion_matrix.sum(axis=1)[i]
    recall = confusion_matrix[i][i] / confusion_matrix.sum(axis=0)[i]
    f1_scores[i] = 2 * precision * recall / (precision + recall)
    
print (f1_scores)
print (f1_scores.mean())


# In[57]:


# PART B
accuracy_random = 0
accuracy_majority = 0

# Get accuracy of random prediction over test set
for test_example in test_data:
    stars = random.randint(1,5)
    if stars == test_example["stars"]:
        accuracy_random += 1
        
# Get accuracy of majority prediction over test set
stars_count = [0] * 5
for example in train_data:
    stars_count[int(example["stars"])-1] += 1
max_count_stars = stars_count.index(max(stars_count)) + 1

for test_example in test_data:
    if max_count_stars == test_example["stars"]:
        accuracy_majority += 1
        
print (accuracy_random/test_m * 100)
print (accuracy_majority/test_m * 100)

