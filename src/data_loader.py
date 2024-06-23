import pickle
import gzip

import numpy as np

def load_data():
    f = gzip.open("/Users/harishprabhu/Documents/GitHub/Neural_Network_from_Scratch/data/mnist.pkl.gz", 'rb')
    u = pickle._Unpickler(f)
    u.encoding = 'latin1'
    training_data, validation_data, test_data = u.load()
    f.close()
    return (training_data, validation_data, test_data)


def load_data_wrapper():

    tr_d, val_d, test_d = load_data()
    training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
    training_results = [vectorized_result(y) for y in tr_d[1]]
    training_data = list(zip(training_inputs, training_results))
    validation_inputs = [np.reshape(x, (784, 1)) for x in val_d[0]]
    validation_results = [vectorized_result(y) for y in val_d[1]]
    validation_data = list(zip(validation_inputs, validation_results))
    test_inputs = [np.reshape(x, (784, 1)) for x in test_d[0]]
    test_results = [vectorized_result(y) for y in test_d[1]]
    testing_data = list(zip(test_inputs, test_results))
    return training_data, validation_data, testing_data


def vectorized_result(j):
    """
    This is also the process of performing ONE HOT ENCODING on all the labels.
    """
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e