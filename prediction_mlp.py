import os
import sys
import glob
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

def load_data(source_dir='./data/final_project'):
    
    configs = []
    learning_curves = []
    
    for fn in glob.glob(os.path.join(source_dir, "*.json")):
        with open(fn, 'r') as fh:
            tmp = json.load(fh)
            configs.append(tmp['config'])
            learning_curves.append(tmp['learning_curve'])
    return(configs, learning_curves)

configs, learning_curves = load_data()

# keep a copy for input and output labels
configs_copy = configs
learning_curves_copy = learning_curves

N = len(configs)
#print(N)
n_epochs = len(learning_curves[0])

configs_df = pd.DataFrame(configs)
learning_curves = np.array(learning_curves)

labels = np.zeros((N, 1))
for i in range(N):
    line = learning_curves[i]
    labels[i][0] = line[-1]
    

inputs = np.zeros((N, 5))
for i in range(N):
    json_in = configs[i]
    inputs[i][0] = json_in['batch_size']
    inputs[i][1] = json_in['log10_learning_rate']
    inputs[i][2] = json_in['log2_n_units_1']
    inputs[i][3] = json_in['log2_n_units_2']
    inputs[i][4] = json_in['log2_n_units_3']
    
n_subset=20
t_idx = np.arange(1, n_epochs+1)

[plt.plot(t_idx, lc) for lc in learning_curves[:n_subset]]
plt.title("Subset of learning curves")
plt.xlabel("Number of epochs")
plt.ylabel("Validation error")
plt.show()

sorted = np.sort(learning_curves[:, -1])
h = plt.hist(sorted, bins=20)
plt.show()

yvals = np.arange(len(sorted))/float(len(sorted))
plt.plot(sorted, yvals)
plt.title("Empirical CDF")
plt.xlabel("y(x, t=40)")
plt.ylabel("CDF(y)")
plt.show()

all_values = np.sort(learning_curves.flatten())

h = plt.hist(all_values, bins=20)
plt.show()

yvals = np.arange(all_values.shape[0])/all_values.shape[0]
plt.plot(all_values, yvals)
plt.title("Empirical CDF")
plt.xlabel("y(x, t=40)")
plt.ylabel("CDF(y)")
plt.show()

import torch
import torch.nn as nn
import torchvision
import numpy as np
from torch.autograd import Variable
from sklearn import preprocessing
# Hyperparameters

input_size = 5 # 5 features
hidden_size = 64
output_size = 1 # 
num_epochs = 2000
batch_size = 100
learning_rate = 0.0001

#expontial learning rate
class Step(object):
    def lr(lr_0, stepsize, nepochs):
        lr_list = []
        for i in range(nepochs):
            lr = lr_0 * np.power(0.95, np.floor(i/stepsize))
            lr_list.append(lr)
        return lr_list
    

# Neural Network Model (2 hidden layers MLP)
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, drop_rate):
        super(MLP, self).__init__()
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        #dropout layer
        self.drop_rate = drop_rate
        self.drop = nn.Dropout(p=drop_rate)
        #self.lr_schedule = lr_schedule
    
    def forward(self, x):
        out = self.drop(x)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.drop(out)
        out = self.fc2(out)        
        out = self.relu(out)
        out = self.drop(out)
        out = self.fc3(out)
        return out





def exp_lr_scheduler(optimizer, epoch, init_lr=0.0001, lr_decay_epoch=100):
    """Decay learning rate by a factor  http://pytorch.org/docs/master/optim.html#how-to-adjust-learning-rate"""
    lr = init_lr * np.power(0.95, np.floor(epoch/lr_decay_epoch))
    #lr = init_lr
 #   print('Learning Rate: %.5f' %(lr))
    #print(np.power(0.5, np.floor(epoch/lr_decay_epoch))*init_lr)
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        
    return optimizer
    
#Train the model
#train_input
#train_input = (([fea1,...,fea5],[label1]),([fea1,..,fea5],[label2]), ... ([])  );
#train_input = ( []   );
#inputs = array([5*256])
#outputs = array([1*256])  

# pre-process training input and labels
inputs_torch = torch.from_numpy(inputs).float()
inputs_variable = Variable(inputs_torch)
        
labels_torch = torch.from_numpy(labels).float()
labels_variable = Variable(labels_torch, requires_grad = False)

"""
for epoch in range(num_epochs):
    
    
        # Convert torch tensor to Variable
        

        
        # Forward + Backward + Optimize
        optimizer.zero_grad() # zero the gradient
        outputs = mlp(inputs_variable)
        #print (type(outputs), type(inputs), type(labels))
        #print (inputs)
 #       print (outputs)
  #      print (labels)
        loss = criterion(outputs, labels_variable)
        loss.backward()
        optimizer.step()
        optimizer = exp_lr_scheduler(optimizer, epoch)
        
        print ('Epoch [%d/%d], Loss: %.4f' 
               %(epoch+1, num_epochs, loss.data[0]))
"""

"""
shuffle the input and output labels and use 3-folder cross validation
"""
def randomize(inputs, lables):
    # Generate the permutation index array
    permutation = np.random.permutation(inputs.shape[0])
    shuffled_inputs = inputs[permutation]
    shuffled_labels = lables[permutation]
    return shuffled_inputs, shuffled_labels

#shuffled_inputs, shuffled_labels = randomize(inputs, labels) 

# divide into 3 folders
N = 265
K = np.int(N/3)
kf = KFold(n_splits= 3)

total_loss = np.array([0.,0.])# altogether 2 different settings for hyperparameter
print (total_loss)

# The first hyparameter setting
for train, test in kf.split(labels): 
    # New training data and test data after cross validation split
    train_inputs, test_inputs, train_labels, test_labels = inputs[train], inputs[test], labels[train], labels[test]     

    # Feature prerpocessing
    scaler = preprocessing.StandardScaler().fit(train_inputs)
    scaler.transform(train_inputs)
    scaler.transform(test_inputs)
    learning_rate = 0.001
    # Loss and Optimer
    mlp = MLP(input_size, hidden_size, output_size, drop_rate = 0.2)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(mlp.parameters(), lr=learning_rate, weight_decay=0.01)
    
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        
        #torch to variable
        train_inputs_variable = Variable(torch.from_numpy(train_inputs).float())
        train_labels_variable = Variable(torch.from_numpy(train_labels).float())
        
        test_inputs_variable = Variable(torch.from_numpy(test_inputs).float())
        test_labels_variable = Variable(torch.from_numpy(test_labels).float())
        
        # train model
        mlp.train()
        ouputs = mlp(train_inputs_variable)
        loss = criterion(ouputs, train_labels_variable)
        loss.backward()
        optimizer.step()
        optimizer = exp_lr_scheduler(optimizer,epoch)
        
        if epoch%100 == 0:
            print ('Epoch [%d/%d], Traing Loss: %.4f' 
                   %(epoch+1, num_epochs, loss.data[0]))
    
    # test model
    mlp.eval() # for the case of dropout-layer and Batch-normalization
    outputs = mlp(test_inputs_variable)
    test_loss = criterion(outputs, test_labels_variable)
    print (test_loss.data[0])
    total_loss[0] += test_loss.data[0]
    print (total_loss)

    

# The 2nd hyparameter setting
for train, test in kf.split(labels): 
    # New training data and test data after cross validation split
    train_inputs, test_inputs, train_labels, test_labels = inputs[train], inputs[test], labels[train], labels[test]     

    # Feature prerpocessing
    scaler = preprocessing.StandardScaler().fit(train_inputs)
    scaler.transform(train_inputs)
    scaler.transform(test_inputs)
    
    learning_rate = 0.001
    # Loss and Optimer
    mlp = MLP(input_size, hidden_size, output_size, drop_rate = 0.2)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(mlp.parameters(), lr=learning_rate, weight_decay=0.1)
    
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        
        #torch to variable
        train_inputs_variable = Variable(torch.from_numpy(train_inputs).float())
        train_labels_variable = Variable(torch.from_numpy(train_labels).float())
        
        test_inputs_variable = Variable(torch.from_numpy(test_inputs).float())
        test_labels_variable = Variable(torch.from_numpy(test_labels).float())
        
        # train model
        mlp.train()
        ouputs = mlp(train_inputs_variable)
        loss = criterion(ouputs, train_labels_variable)
        loss.backward()
        optimizer.step()
        optimizer = exp_lr_scheduler(optimizer,epoch)
        
        if epoch%100 == 0:
            print ('Epoch [%d/%d], Traing Loss: %.4f' 
                   %(epoch+1, num_epochs, loss.data[0]))
    
    # test model
    mlp.eval()
    outputs = mlp(test_inputs_variable)
    test_loss = criterion(outputs, test_labels_variable)
    print (test_loss.data[0])
    total_loss[1] += test_loss.data[0]
    print (total_loss)

