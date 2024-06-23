# import numpy as np
# import random 
# import time

# class Network(object):
    
#     def __init__(self, sizes):
#         self.num_layers = len(sizes)
#         self.sizes = sizes
#         self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
#         self.weights = [np.random.radn(y, x) 
#                     for x, y in zip(sizes[:-1], sizes[1:])]
        

#     def feed_forward(self, a):
#         """Returns the output of network if "a" is the input."""
#         """It is assumed that the input "a" is an (n, 1) numpy ndarray, not (n, ) vector. This is FLATTENING and then fed in to the FC Neural Network """
#         for b, w in zip(self.biases, self.weights):
#             a = sigmoid(np.dot(w, a) + b)



# ## Creating Sigmoid Activation Function
# def sigmoid(z):
#     return 1.0/(1.0 + np.exp(-z))






