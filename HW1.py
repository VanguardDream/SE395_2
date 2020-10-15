import sys
import os

import time
import numpy as np
import matplotlib.pyplot as plt
import scipy
from PIL import Image
from scipy import ndimage

import functions
import Loader
import network as nw
import dnn_app_utils_v2 as util

macos = False

codePath = os.path.dirname( os.path.abspath("HW1.py"))

if not macos:
    fp_trainImage = open(codePath+'\\data\\train\\train-images.idx3-ubyte','rb')
    fp_trainLabel = open(codePath+'\\data\\train\\train-labels.idx1-ubyte','rb')

    fp_testImage = open(codePath+'\\data\\test\\test-images.idx3-ubyte','rb')
    fp_testLabel = open(codePath+'\\data\\test\\test-labels.idx1-ubyte','rb')
else :
    fp_trainImage = open(codePath+'/data/train/train-images.idx3-ubyte','rb')
    fp_trainLabel = open(codePath+'/data/train/train-labels.idx1-ubyte','rb')

    fp_testImage = open(codePath+'/data/test/test-images.idx3-ubyte','rb')
    fp_testLabel = open(codePath+'/data/test/test-labels.idx1-ubyte','rb')

# Jump MNIST file header
tmp = fp_trainImage.read(16)
tmp = fp_trainLabel.read(8)
tmp = fp_testImage.read(16)
tmp = fp_testLabel.read(8)
# -----------------------------------

train_x, train_y, classes, EOF = Loader.load(1, fp_trainImage, fp_trainLabel)
test_x, test_y, classes, EOF = Loader.load(1, fp_testImage, fp_testLabel)


# while True:
#     train_x, train_y, classes, EOF = Loader.load(100, fp_trainImage, fp_trainLabel)
#     if EOF:
#         break

# sys.exit()

###
lamda = 0.005 #Learning rate
iteration = 2500
###
W1, b1, W2, b2, W3, b3 = functions.initialize_parameters(h1=128,h2=64)

for i in range(0, iteration):
    A1, cache_1 = functions.linear_propagate(train_x, W1, b1, 'relu')
    A2, cache_2 = functions.linear_propagate(A1, W2, b2, 'relu')
    Y_hat, cache_3 = functions.linear_propagate(A2, W3, b3, 'softmax')

    cost = functions.compute_cost(Y_hat, train_y)
    pb = functions.accuracy(Y_hat, train_y)
    
    print(cost)



    # dA3 = -1 * (np.divide(train_y, Y_hat) - np.divide(1 - train_y, 1 - Y_hat))
    dA3 = Y_hat - train_y

    dA2, dW3, db3 = functions.linear_backpropagate(dA3, cache_3, 'softmax')
    dA1, dW2, db2 = functions.linear_backpropagate(dA2, cache_2, 'relu')
    dA0, dW1, db1 = functions.linear_backpropagate(dA1, cache_1, 'relu')

    #Update W
    W1 = W1 - (lamda * dW1)
    W2 = W2 - (lamda * dW2)
    W3 = W3 - (lamda * dW3)

    #Update b
    b1 = b1 - (lamda * db1)
    b2 = b2 - (lamda * db2)
    b3 = b3 - (lamda * db3)

A1, cache_1 = functions.linear_propagate(train_x, W1, b1, 'relu')
A2, cache_2 = functions.linear_propagate(A1, W2, b2, 'relu')
Y_hat, cache_3 = functions.linear_propagate(A2, W3, b3, 'softmax')

print(Y_hat)

sys.exit()



