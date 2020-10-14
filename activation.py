import numpy

def sigmoid (x):
    A = 1 / (1 + numpy.exp(-x))
    return A

def relu (x):
    return numpy.maximum(0,x)

def lrelu (x, slope = 0.2):
    A = numpy.max(x * 0.1, x)
    return A

def softmax (x):
    exps = numpy.exp(x)
    A = exps / numpy.sum(exps)
    return A

def sigmoid_back(dA, cache):    
    Z = cache
    
    s = 1/(1+numpy.exp(-Z))
    dZ = dA * s * (1-s)
    
    assert (dZ.shape == Z.shape)
    
    return dZ

def relu_back(dA, cache):    
    Z = cache

    dZ = numpy.array(dA, copy=True) # just converting dz to a correct object.
    
    # When z <= 0, you should set dz to 0 as well. 
    dZ[Z <= 0] = 0
    
    assert (dZ.shape == Z.shape)
    
    return dZ

def lrelu_back(dA,cache,slope = 0.2):
    Z = cache

    dZ = numpy.array(dA, copy=True)

    dZ[Z <= 0] = slope

    assert (dZ.shape == Z.shape)

    return dZ

def softmax_back(dA, cache, Y):
    """
    cache(Z) = prob
    Y = label
    """
    Z = cache

    dZ = Z * (1 - Z)

    assert (dZ.shape == Z.shape)

    return dZ