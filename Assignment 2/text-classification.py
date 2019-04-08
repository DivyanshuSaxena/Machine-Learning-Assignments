import json
import nltk
import math
import random
import re
import time
import sys
import numpy as np
from nltk.tokenize import RegexpTokenizer, TweetTokenizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk import bigrams

# Download Stopwords
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
    
# tokenizer = RegexpTokenizer(r'\w+')
tokenizer = TweetTokenizer()
stop_words = list(set(stopwords.words('english')))

# Hyper parameters
profile = False

# Inputs
train_file = sys.argv[1]
test_file = sys.argv[2]
part = sys.argv[3]

# Parameters for PART D
use_stop_words = part >= 'd'
use_stemming = part == 'd'
use_doc_freq = part >= 'e'

# Parameters for PART E
use_bigrams = part >= 'e'
if use_bigrams:
    punctuation = ['.', ',', ':', '(', ')', '?', '<', '?', '>', '\'', '\"']
    stop_words.extend(punctuation)
ps = PorterStemmer()

train_data = []
test_data = []

with open(train_file) as f:
    train_data = json.load(f)
        
with open(test_file) as f:
    test_data = json.load(f)


if part != 'b':
    vocab = {}
    word_index = 0
    m = len(train_data)

    def get_vocab():
        global word_index
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
                    vocab[word] = (word_index, 1)
                    word_index += 1
                else:
                    vocab[word] = (vocab[word][0], vocab[word][1]+1)
                    
            # Add bigrams in vocab
            if use_bigrams:
                bigrams = list(nltk.bigrams(review))
                # print (bigrams)
                for bigram in bigrams:
                    insert_bigram = bigram[0] + "_" + bigram[1] 
                    if insert_bigram not in vocab:
                        vocab[insert_bigram] = (word_index, 1)
                        word_index += 1
                    else:
                        vocab[insert_bigram] = (vocab[insert_bigram][0], vocab[insert_bigram][1]+1)    
                        
            if word_index%10000 == 0:
                print ("Added {0} words in dictionary".format(word_index))
        try:    
            vocab_text = open('vocab.txt', 'w')
            for k, v in vocab.items():
                vocab_text.write("{0} {1}\n".format(k, v[1]))
            vocab_text.close()
            print (len(vocab.keys()))
        except:
            print (len(vocab.keys()))

    try:
        with open('/vocab.txt') as f:
            temp_vocab = f.readlines()
            for word_tuple in temp_vocab:
                word_tuple = word_tuple.rstrip()
                split_index = word_tuple.rindex(' ')
                word = word_tuple[0:split_index]
                occurence = int(word_tuple[split_index+1:])
                # print ("{0} {1}".format(word, occurence))
                vocab[word] = (word_index, occurence)
                word_index += 1
            f.close()
        print (len(vocab))
    except FileNotFoundError:
        get_vocab()



    # Constants for the algorithm
    V = len(vocab.keys())

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
                t = vocab[word]
                # print ("Found word {0} at {1}".format(word, t))
                if ((not use_doc_freq) or (use_doc_freq and t[1] >= 2 and t[1] <= m/2)):
                    thetas[(stars-1)*V + t[0]] += 1
            if use_bigrams:
                bigrams = list(nltk.bigrams(review))
                lengths[stars-1] += len(bigrams)
                for bigram in bigrams:
                    final_bigram = bigram[0]+'_'+bigram[1]
                    t = vocab[final_bigram]
                    if ((not use_doc_freq) or (use_doc_freq and t[1] >= 2 and t[1] <= m/2)):
                        thetas[(stars-1)*V + t[0]] += 1
                        # print (bigram[0]+'_'+bigram[1], k)
        
    evaluate_thetas()



    # Make final evaluations for the parameters
    log_thetas = [0] * (V*5)
    for k in range(V*5):
        star_index = int(k/V)
        log_thetas[k] = math.log(thetas[k]/lengths[star_index])
        
    for c in range(5):
        phis[c] = counts[c]/m



    test_m = len(test_data)
    confusion_matrix = np.zeros(shape=(5,5))

    def get_accuracy():
        # Get the accuracy of the trained model over the training data and test data for PART A
        accuracy = 0
        for example in test_data[0:test_m]:
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
                        t = vocab[word]
                        # print ("Found word {0} at {1}".format(word, t))
                        if ((not use_doc_freq) or (use_doc_freq and t[1] >= 2 and t[1] <= m/2)):
                            ll += log_thetas[V*(y-1) + t[0]]
                    except:
                        ll += math.log(1/V)
                if use_bigrams:
                    bigrams = list(nltk.bigrams(review))
                    for bigram in bigrams:
                        try:
                            t = vocab[bigram[0]+'_'+bigram[1]]
                            if ((not use_doc_freq) or (use_doc_freq and t[1] >= 2 and t[1] <= m/2)):
                                ll += log_thetas[V*(y-1) + t[0]]
                        except:
                            ll += math.log(1/V)
                # print ("Likelihood: {0} for {1} stars".format(ll, y))
                if (max_ll == 0 or max_ll < ll):
                    max_ll = ll
                    max_ll_stars = y

            # Calculated the max likelihood stars. Now check accuracy
            # print ("{0} {1}".format(max_ll_stars, example["stars"]))
            if max_ll_stars == example["stars"]:
                accuracy += 1
                
            # Evaluate confusion matrix for PART C
            confusion_matrix[max_ll_stars-1][int(example["stars"])-1] += 1

        print (accuracy/test_m * 100)
        print (np.matrix(confusion_matrix))
        
    get_accuracy()


    if part >= 'e':
        # Calculate F1 Scores for PART F
        f1_scores = np.zeros(5)
        for i in range(5):
            precision = confusion_matrix[i][i] / confusion_matrix.sum(axis=1)[i]
            recall = confusion_matrix[i][i] / confusion_matrix.sum(axis=0)[i]
            f1_scores[i] = 2 * precision * recall / (precision + recall)
            
        print (f1_scores)
        print (f1_scores.mean())

else:
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

