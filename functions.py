import numpy as np
import activation

def initialize_parameters(n_x = 784, h1 = 64 , h2 = 32, n_y = 10):
    """
    n_x : size of input (n_x, number of images)
    h1  : number of layers (1)
    h2  : number of layers (2)
    """
    np.random.seed(1)
    
    W1 = np.random.randn(h1, n_x)*0.01
    b1 = np.zeros((h1, 1))
    W2 = np.random.randn(h2, h1)*0.01
    b2 = np.zeros((h2, 1))
    W3 = np.random.randn(10, h2)*0.01
    b3 = np.zeros((10,1))
    
    assert(W1.shape == (h1, n_x))
    assert(b1.shape == (h1, 1))
    assert(W2.shape == (h2, h1))
    assert(b2.shape == (h2, 1))
    assert(W3.shape == (10, h2))
    assert(b3.shape == (10,1))
    
    return W1, b1, W2, b2, W3, b3  

def linear_propagate(A_prev, W, b, activate):

    linear_cache = (A_prev, W, b)

    if activate == "sigmoid":
        Z = np.dot(W, A_prev) + b
        activation_cache = Z
        A = activation.sigmoid(Z)
    
    elif activate == "relu":
        Z = np.dot(W, A_prev) + b
        activation_cache = Z
        A = activation.relu(Z)

    elif activate == "softmax":
        Z = np.dot(W, A_prev) + b
        activation_cache = Z
        A = activation.softmax(Z)
    elif activate == "lrelu":
        Z = np.dot(W, A_prev) + b
        activation_cache = Z
        A = activation.lrelu(Z)
    
    assert (A.shape == (W.shape[0], A_prev.shape[1]))

    cache = (linear_cache, activation_cache)

    return A, cache

def linear_backpropagate(dA, cache, activate):
    """
    return dA_prev, dW, db
    """
    linear_cache, activation_cache = cache

    A_prev, W, b = linear_cache
    m = A_prev.shape[1]

    if activate == "relu":
        dZ = activation.relu_back(dA, activation_cache)
        dA_prev = np.dot(W.T,dZ)
        dW = (1/m) * np.dot(dZ, A_prev.T)
        db = (1/m) * np.sum(dZ, axis = 1, keepdims = True)
    elif activate == "relu":
        dZ = activation.relu_back(dA, activation_cache)
        dA_prev = np.dot(W.T,dZ)
        dW = (1/m) * np.dot(dZ, A_prev.T)
        db = (1/m) * np.sum(dZ, axis = 1, keepdims = True)  
    elif activate == "sigmoid":
        dZ = activation.sigmoid_back(dA, activation_cache)
        dA_prev = np.dot(W.T,dZ)
        dW = (1/m) * np.dot(dZ, A_prev.T)
        db = (1/m) * np.sum(dZ, axis = 1, keepdims = True)
    elif activate == "softmax":
        dZ = activation.softmax_back(dA, activation_cache)
        dA_prev = np.dot(W.T,dZ)
        dW = (1/m) * np.dot(dZ, A_prev.T)
        db = (1/m) * np.sum(dZ, axis = 1, keepdims = True)
    
    return dA_prev, dW, db


def compute_cost(Y_hat, Y): 
    m = Y.shape[1]

    # s = -(Y) * np.log(Y_hat)
    # cost = (1/m) * np.sum(s)

    cost = (-1 / m) * np.sum(np.multiply(Y, np.log(Y_hat)) + np.multiply(1 - Y, np.log(1 - Y_hat)))

    # loss = np.argmax(Y_hat) - np.argmax(Y)
    # cost = (1/m) * np.sum(loss)
    
    return cost

def accuracy(Y_hat, y):
    return (Y_hat.argmax() == y.astype('float32')).mean()

