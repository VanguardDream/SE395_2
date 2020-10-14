import sys
import os

import time
import numpy as np
import matplotlib.pyplot as plt
import scipy
from PIL import Image
from scipy import ndimage

import Loader
import network as nw
import dnn_app_utils_v2 as util

macos = True

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

train_x, train_y, classes = Loader.load(50, fp_trainImage, fp_trainLabel)
test_x, test_y, classes = Loader.load(50, fp_testImage, fp_testLabel)

a = np.reshape([0, 0.1, 0.9],(3,1))
b = np.reshape([0, 0, 1],(3,1))

print(util.compute_cost(a,b))
sys.exit()

n_x = 28*28     # num_px * num_px * 3
n_h = 2
n_y = 10
layers_dims = (n_x, n_h, n_y)

parameters = nw.two_layer_model(train_x, train_y, layers_dims = (n_x, n_h, n_y), num_iterations = 2800, print_cost=True)

pred_train = util.predict(train_x, train_y, parameters)