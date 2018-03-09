import os
import glob
import json
import random
import numpy as np
from random import shuffle
from sklearn import preprocessing

# raw load data function
def load_data(source_dir='./data/final_project'):
    configs = []
    learning_curves = []

    for fn in glob.glob(os.path.join(source_dir, "*.json")):
        with open(fn, 'r') as fh:
            tmp = json.load(fh)
            configs.append(tmp['config'])
            learning_curves.append(tmp['learning_curve'])
    return (configs, learning_curves)

# shuffle input and output data，this process is
def randomize(inputs, labels):
    random.seed(10)
    n = len(inputs)
    index = [i for i in range(n)]
    shuffle(index)
    inputs = inputs[index]
    labels = labels[index]
    return (inputs, labels)


# load data with array format
def load_data_raw():
    configs, learning_curves = load_data()

    N = len(configs)
    labels = np.zeros((N, 1))
    for i in range(N):
        labels[i][0] = learning_curves[i][-1]

    inputs = np.zeros((N, 5))
    for i in range(N):
        inputs[i][0] = configs[i]['batch_size']
        inputs[i][1] = configs[i]['log10_learning_rate']
        inputs[i][2] = configs[i]['log2_n_units_1']
        inputs[i][3] = configs[i]['log2_n_units_2']
        inputs[i][4] = configs[i]['log2_n_units_3']

    #inputs, labels = randomize(inpus, labels)

    return inputs, labels


#load data where every input dimension has zero mean and unit variance
def load_data_standardization():

    inputs, labels = load_data_raw()
    inputs = preprocessing.scale(inputs)

    return(inputs, labels)


def load_data_learning_curve(n_epochs=5):
    configs, learning_curves = load_data()

    N = len(configs)

    inputs = np.zeros((N, n_epochs))
    for i in range(N):
        inputs[i] = learning_curves[i][:n_epochs]

    labels = np.zeros((N, 1))
    for i in range(N):
        labels[i][0] = learning_curves[i][-1]

    configuration = np.zeros((N, 5))
    for i in range(N):
        configuration[i][0] = configs[i]['batch_size']
        configuration[i][1] = configs[i]['log10_learning_rate']
        configuration[i][2] = configs[i]['log2_n_units_1']
        configuration[i][3] = configs[i]['log2_n_units_2']
        configuration[i][4] = configs[i]['log2_n_units_3']

    return(np.concatenate((configuration, inputs), axis=1), labels)

def load_data_four_points():
    configs, learning_curves = load_data()

    N = len(configs)
    inputs = np.zeros((N*36, 9))
    labels = np.zeros((N*36, 1))

    b = 0
    for i in range(N*36-1):
        a = int((i+1)/36)

        inputs[i][0] = configs[a]['batch_size']
        inputs[i][1] = configs[a]['log10_learning_rate']
        inputs[i][2] = configs[a]['log2_n_units_1']
        inputs[i][3] = configs[a]['log2_n_units_2']
        inputs[i][4] = configs[a]['log2_n_units_3']
        inputs[i][5] = learning_curves[a][b]
        inputs[i][6] = learning_curves[a][b+1]
        inputs[i][7] = learning_curves[a][b+2]
        inputs[i][8] = learning_curves[a][b+3]
        labels[i] = learning_curves[a][b+4]

        b = 0 if b==35 else b+1

    return(inputs, labels)



