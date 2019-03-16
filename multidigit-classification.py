import math
import random
import time
import svmutil
import sys
import numpy as np
import matplotlib.pyplot as plt
from cvxopt import matrix
from cvxopt import solvers
from cvxopt.solvers import qp

# Hyperparameters
gamma = 0.05
gaussian = True

# Inputs
train_file = sys.argv[1]
test_file = sys.argv[2]
part = sys.argv[3]

# Get data for PART A
train_data_raw = np.genfromtxt(train_file, delimiter=',')
test_data_raw = np.genfromtxt(test_file, delimiter=',')

# Normalize data and divide into arrays
scale_down = lambda x : x/255
train_data = []
test_data = []

for i in range(10):
    train_data.append([])
    test_data.append([])

for sample in train_data_raw:
    y = int(sample[-1])
    insert_sample = list(map(scale_down, sample[:-1]))
    insert_sample.append(sample[-1])
    train_data[y].append(insert_sample)
    
for sample in test_data_raw:
    y = int(sample[-1])
    insert_sample = list(map(scale_down, sample[:-1]))
    insert_sample.append(sample[-1])
    test_data[y].append(insert_sample)
    
solvers.options['show_progress'] = False

collated_test = np.array([])
collated_train = np.array([])
for i in range(10):
    if i == 0:
        collated_test = test_data[i]
        collated_train = train_data[i]
    else:
        collated_test = np.concatenate((collated_test, test_data[i]), axis=0)
        collated_train = np.concatenate((collated_train, train_data[i]), axis=0)

test_m = len(collated_test)
train_m = len(collated_train)

if part == 'a' or part == 'c':
    def get_parameters(digit1, digit2):
        get_class = lambda x : 1 if x == digit1 else -1

        partial_data = np.concatenate((train_data[digit1], train_data[digit2]), axis=0)
        m = len(partial_data)
        alphas = np.array([])
        w = np.array([])
        b = 0

        # Get input for the solver
        X = np.delete(partial_data, -1, axis=1)
        if not gaussian:
            Y = np.diag([get_class(y) for y in partial_data[:, -1]])
            kernel = np.matmul(X, X.T)
            temp_P = np.matmul(np.matmul(Y, kernel), Y)
            P = matrix(temp_P)
        else:
            Y = np.diag([get_class(y) for y in partial_data[:, -1]])
            xtx = np.sum(np.multiply(X, X), 1).reshape(m, 1)
            kernel_noexp = xtx + xtx.T - 2 * np.dot(X, X.T)
            kernel = np.power(np.exp(-1*gamma), kernel_noexp)
            temp_P = np.matmul(np.matmul(Y, kernel), Y)
            P = matrix(temp_P)

        q = matrix(1.0, (m,1))

        G = matrix(np.identity(m))
        G_identity = np.identity(m)
        temp_G = np.concatenate((G_identity, -G_identity), axis=0)
        G = matrix(temp_G)

        h = matrix(0.0, (m,1))
        h_zero = np.zeros(m)
        h_ones = np.ones(m)
        temp_h = np.append(h_zero, h_ones)
        h = matrix(temp_h, (2*m,1))

        temp_A = list(map(get_class, partial_data[0:m, -1]))
        A = matrix(np.array(temp_A), (1, m), 'd')

        b = matrix(0.0)

        # Use the cvxopt solver qp module
        alphas = qp(P, q, G, h, A, b)['x']
        alphas = np.array(-alphas)[:, 0]
        print (alphas)
        
        # Evaluate w if linear kernel used
        if not gaussian:
            w = np.zeros(28*28)
            for i in range(m):
                sample = partial_data[i]
                w += alphas[i] * get_class(sample[-1]) * sample[:-1]

        # Evaluate b
        alpha_y = np.multiply(alphas, np.array(list(map(get_class, partial_data[:, -1]))) )
        print (alpha_y.shape)

        w_trans_X = np.matmul(kernel, alpha_y)
        maxone = -99999999
        minone = 99999999
        for i in range(m):
            wtx = w_trans_X[i]
            y = get_class(partial_data[i][-1])
            if y == -1:
                maxone = max(maxone, wtx)
            else:
                minone = min(minone, wtx)

        b = -(maxone + minone)/2
        print (b)
        
        return alpha_y, w, b

    w = []
    alpha_y = []
    partial_data = []
    X = []
    XtX = []
    Y = []
    b = np.zeros(shape=(10, 10))
    lengths = []
    for i in range(10):
        row_w = []
        row_alpha = []
        row_data = []
        row_x = []
        row_y = []
        row_lengths = []
        row_xtx = []
        for j in range(10):
            row_w.append(np.array([]))
            row_alpha.append(np.array([]))
            row_data.append(np.array([]))
            row_x.append(np.array([]))
            row_y.append(np.array([]))
            row_xtx.append(np.array([]))
            row_lengths.append(0)
        w.append(row_w)
        alpha_y.append(row_alpha)
        partial_data.append(row_data)
        X.append(row_x)
        Y.append(row_y)
        lengths.append(row_lengths)
        XtX.append(row_xtx)

    get_class = lambda x, digit : 1 if x == digit else -1
        
    # Collate Partial Data
    for digit1 in range(0, 9):
        for digit2 in range(digit1+1, 10):
            partial_data[digit1][digit2] = np.concatenate((train_data[digit1], train_data[digit2]), axis=0)
            X[digit1][digit2] = np.delete(partial_data[digit1][digit2], -1, axis=1)                
            Y[digit1][digit2] = np.diag([get_class(y, digit1) for y in partial_data[digit1][digit2][:, -1]])
            lengths[digit1][digit2] = len(partial_data[digit1][digit2])
            XtX[digit1][digit2] = np.sum(np.multiply(X[digit1][digit2], X[digit1][digit2]), 1).reshape(lengths[digit1][digit2], 1)


    # Obtain the classifiers for each pair of digits
    def get_parameters_from_file():
        try:
            with open('parameters.txt') as f: 
                parameters = [x.rstrip() for x in f.readlines()]
                counter = 0
                for digit1 in range(9):
                    for digit2 in range(digit1+1, 10):
                        alpha_y[digit1][digit2] = np.fromstring(parameters[counter], dtype=float, sep=',')
                        b[digit1][digit2] = float(parameters[counter+1])
                        counter += 2
        except FileNotFoundError:
            for digit1 in range(10):
                for digit2 in range(digit1+1, 10):
                    t = get_parameters(digit1, digit2)
                    alpha_y[digit1][digit2], w[digit1][digit2], b[digit1][digit2] = t[0], t[1], t[2]
                    print ("Done for {0}, {1}".format(digit1, digit2))
                    w[digit2][digit1] = w[digit1][digit2]
                    b[digit2][digit1] = b[digit1][digit2]
            # Write into file
            parameters_text = open('parameters.txt', 'w')
            for digit1 in range(9):
                for digit2 in range(digit1+1, 10):
                    alpha_str = ', '.join("{0:.10f}".format(x) for x in alpha_y[digit1][digit2]) # '0,3,5'
                    parameters_text.write(alpha_str)
                    parameters_text.write("\n{0}\n".format(b[digit1][digit2]))
            parameters_text.close()

    get_parameters_from_file()

    def get_accuracy(data, length):
        accuracy = counter = 0
        confusion_matrix = np.zeros(shape=(10,10))
        for sample in data[0:1000]:
            counts = [(0,0)] * 10
            counter += 1
            xtx = np.sum(np.multiply(sample[:-1], sample[:-1])).reshape(1,1)
            for digit1 in range(0, 9):
                for digit2 in range(digit1+1, 10):
                    b_local = b[digit1][digit2]
                    if not gaussian:
                        w_local = w[digit1][digit2]
                        pred_z = np.dot(w_local.T, sample[:-1]) + b_local
                    else:
                        inner_product = xtx + XtX[digit1][digit2].T - 2 * np.dot(sample[:-1], X[digit1][digit2].T)
                        wtx = np.dot(alpha_y[digit1][digit2], np.power(np.exp(-gamma), inner_product.T))
                        pred_z = wtx + b_local

                    if pred_z > 0:
                        counts[digit1] = (counts[digit1][0]+1, counts[digit1][1]+pred_z)
                        # print("Predicting {0} among {0},{1}".format(digit1, digit2))
                    else:
                        counts[digit2] = (counts[digit2][0]+1, counts[digit2][1]-pred_z)
                        # print("Predicting {1} among {0},{1}".format(digit1, digit2))
            index = max(enumerate(counts), key=lambda x: 1000*x[1][0]+x[1][1])[0]
            # print ([x[0] for x in counts], index)
            if index == sample[-1]:
                accuracy += 1

            # Evaluate confusion matrix for PART C
            confusion_matrix[index][int(sample[-1])] += 1
            ten_percent = 10*length/100
            if counter%length == 0:
                print ("Completed {0}0% with accuracy {1}".format(counter/ten_percent, accuracy/counter))
        print ("Accuracy: ".format(accuracy/length * 100))
        
        # Print confusion matrix for PART C
        np.set_printoptions(suppress=True)
        print (confusion_matrix)

    print ("For Train Data: ", end='')
    get_accuracy(collated_train, train_m)
    print ("For Test Data: ", end='')
    get_accuracy(collated_test, test_m)

elif part == 'b':
    # Use Libsvm
    Y = collated_train[:, -1]
    X = np.delete(collated_train, -1, axis=1)

    start_time = time.time()
    m_gaussian = svmutil.svm_train(Y, X, "-t 2 -c 1 -g 0.05")
    print ("Time taken by LibSVM for training Gaussian Kernels: {0}".format(time.time()-start_time))

    Y_test = collated_test[:, -1]
    X_test = np.delete(collated_test, -1, axis=1)

    print ("For Train Set: ", end='')
    labels_gaussian = svmutil.svm_predict(Y, X, m_gaussian)
    print ("For Test Set: ", end='')
    labels_gaussian = svmutil.svm_predict(Y_test, X_test, m_gaussian)

else:
    # Compute Validation for PART D
    validation_data = np.array([])
    train_validate_data = np.array([])
    for i in range(10):
        length = len(train_data[i])
        validation_index = int(length/10)
        if i == 0:
            validation_data = train_data[i][0:validation_index]
            train_validate_data = train_data[i][validation_index:]
        else:
            validation_data = np.concatenate((validation_data, train_data[i][0:validation_index]), axis=0)
            train_validate_data = np.concatenate((train_validate_data, train_data[i][validation_index:]), axis=0)

    C = [0.00001, 0.001, 1, 5, 10]

    Y_tv = train_validate_data[:, -1]
    X_tv = np.delete(train_validate_data, -1, axis=1)
    Y_validate = validation_data[:, -1]
    X_validate = np.delete(validation_data, -1, axis=1)

    validation_acc = []
    test_acc = []

    for c in C:
        m = svmutil.svm_train(Y_tv, X_tv, "-t 2 -c {0} -g 0.05".format(c))
        p_label, p_acc, p_val = svmutil.svm_predict(Y_validate, X_validate, m)
        p_label_t, p_acc_t, p_val_t = svmutil.svm_predict(Y_test, X_test, m)
        validation_acc.append(p_acc[0])    
        test_acc.append(p_acc_t[0])

    print (validation_acc, test_acc)

    plt.plot(C, validation_acc, 'r-')
    plt.plot(C, test_acc, 'b-')

