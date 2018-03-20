import os
import torch
import torch.nn as nn
import numpy as np
import time
import json
from torch.autograd import Variable
from sklearn.model_selection import KFold
from data import load_data_raw
from data import load_data_standardization
from torch.optim.lr_scheduler import ExponentialLR

import matplotlib.pyplot as plt


input_size = 5
hidden_size = 64
output_size = 1
dr_score = 0.2
learning_rate = 0.01
num_epochs = 40
lr_exponential = 1.0

CONFIG_ID = 1


# Neural Network Model (2 hidden layers MLP)
# construction of the MPL network model
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, drop_rate):
        super(MLP, self).__init__()
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, output_size)
        self.drop = nn.Dropout(p=drop_rate)
        # self.lr_schedule = lr_schedule

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.drop(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.drop(out)
        out = self.fc3(out)
        out = self.relu(out)
        out = self.drop(out)
        out = self.fc4(out)
        return out

if __name__ == "__main__":

    print("Loading data from datasets...")
    
    #load data into numpy arrays
    inputs_raw, labels_raw = load_data_raw()
    print("Raw data loaded.")
    
    #load data with zero mean and unit variance
    inputs_pre, labels_pre = load_data_standardization()
    print("Preprocessed data loaded.")
    
    
    #training input and labels for Varible
    
    #######################
    # raw data sets
    # inputs and labels
    #######################
    #convert the raw inputs data to Variable 
    inputs_torch_raw = torch.from_numpy(inputs_raw).float()
    inputs_variable_raw = Variable(inputs_torch_raw)
    
    #convert the raw labels(regression values) data to Variable
    labels_torch_raw = torch.from_numpy(labels_raw).float()
    labels_variable_raw = Variable(labels_torch_raw, requires_grad=False)

    #######################
    # preprocessed data sets
    # inputs and labels
    #######################
    #convert the preprocessed input data to Variable
    inputs_torch_pre = torch.from_numpy(inputs_pre).float()
    inputs_variable_pre = Variable(inputs_torch_pre)
    
    #convert the preprocess labels(regression values) data to Variable
    labels_torch_pre = torch.from_numpy(labels_pre).float()
    labels_variable_pre = Variable(labels_torch_pre, requires_grad=False)
    
    """
    3-folder cross validation
    
    using raw data sets
    
    """
    # divide into 3 folders
    kf_raw = KFold(n_splits=3, shuffle=True)  
    MLP_learning_curves_raw = [] #record the validation error
    subset_nums_raw = [] # number of each dataset in 3-folders
    MLP_learning_avg_curve_raw = [0.0] * num_epochs
    KFolder_set_raw = []
    # learning rates in 3 folds
    
    # calculate the runtime for MLP traing with raw data  
    start_time_raw = time.time()
    
    # train MLP model with raw data
    for train, val in kf_raw.split(labels_raw):
        # Subsets of training data and validation data after cross validation split
        X_train, y_train, X_val, y_val = inputs_raw[train], labels_raw[train], inputs_raw[val], labels_raw[val]
        subset_num = len(y_val)
        #add the number of the validation data in this folder
        subset_nums_raw.append(subset_num)
        
        # Loss and Optimer
        mlp_raw = MLP(input_size, hidden_size, output_size, drop_rate=dr_score)
        criterion = nn.MSELoss()
        # Adam Regularization
        optimizer_raw = torch.optim.Adam(mlp_raw.parameters(), lr=learning_rate, weight_decay=0.01)
        scheduler_raw = ExponentialLR(optimizer_raw, gamma = lr_exponential)
        MLP_learning_curve_raw = []
        
        #training begins and lasts 40 epochs
        for epoch in range(num_epochs):

            optimizer_raw.zero_grad()

            # torch to variable
            X_train_variable = Variable(torch.from_numpy(X_train).float())
            y_train_variable = Variable(torch.from_numpy(y_train).float())

            X_val_variable = Variable(torch.from_numpy(X_val).float())
            y_val_variable = Variable(torch.from_numpy(y_val).float())

            # train model
            mlp_raw.train()
            outputs = mlp_raw(X_train_variable)
            loss = criterion(outputs, y_train_variable)
            loss.backward()
            optimizer_raw.step()
            scheduler_raw.step()
            """
            if epoch % 100 == 0:
                print('Epoch [%d/%d], Validation Loss: %.4f'
                      % (epoch + 1, num_epochs, loss.data[0]))
            """            
            MLP_learning_curve_raw.append(loss.data[0])
        #training ends
        
        KFolder_set_raw.append(len(val))
        # validate model
        mlp_raw.eval()  # for the case of dropout-layer and Batch-normalization
        outputs = mlp_raw(X_val_variable)
        validation_loss = criterion(outputs, y_val_variable)
        
        print(validation_loss.data[0])
        MLP_learning_curves_raw.append(MLP_learning_curve_raw)
        
        #print(validation_loss.data[0])
        print("End of one cross validation subset")
    
    #calculate average validation error
    end_time_raw = time.time() - start_time_raw
    print('Runtime of training: %.4fs' %end_time_raw)
    
    #print(KFolder_set) # 89, 88, 88
    ###
    #print(MLP_learning_avg_curves)
    
    for i in range(len(MLP_learning_curves_raw)):
        for j in range(len(MLP_learning_curves_raw[i])):
            MLP_learning_avg_curve_raw[j] += MLP_learning_curves_raw[i][j] * KFolder_set_raw[i] / len(labels_raw)
            
                
    ### 
    print("Final Error after 40 epochs: %.4f " % MLP_learning_avg_curve_raw[-1])
    #print(MLP_learning_avg_curve_raw[-1])
       
    t_idx = np.arange(1, num_epochs+1)
    
    # all sub curves in 3-folder cross validation 
    [plt.plot(t_idx, lc) for lc in MLP_learning_curves_raw]
    plt.title("Subset of learning curves")
    plt.xlabel("Number of epochs")
    plt.ylabel("Validation error")
    plt.show()
    
    
    [plt.plot(t_idx, MLP_learning_avg_curve_raw)]
    plt.title("cross validation learning curve")
    plt.xlabel("Number of epochs")
    plt.ylabel("Validation error")
    plt.show()

    
    #Histogram and CDF over the final error
    sorted_MLP_learning_curve_raw = np.sort(MLP_learning_avg_curve_raw)
    h = plt.hist(sorted_MLP_learning_curve_raw, bins=20)
    plt.title("Histogram over the final error")
    plt.show()
    
    yvals = np.arange(len(sorted_MLP_learning_curve_raw))/float(len(sorted_MLP_learning_curve_raw))
    plt.plot(sorted_MLP_learning_curve_raw, yvals)
    plt.title("Empirical CDF")
    plt.xlabel("y(x, t=40)")
    plt.ylabel("CDF(y)")
    plt.show()
    
    ############################
    # 3-folder cross validation
    # using preprocessed datasets
    ############################
    # divide into 3 folders
    kf_pre = KFold(n_splits=3, shuffle=True)  
    MLP_learning_curves_pre = [] #record the validation error
    subset_nums_pre = [] # number of each dataset in 3-folders
    MLP_learning_avg_curve_pre = [0.0] * num_epochs
    KFolder_set_pre = []
    epoch_learning_rates_pre = [] # learning rate in 3 folds
    
    ##################################### 
    # calculate the runtime for MLP traing 
    # with preprocessed data 
    #####################################
    start_time_pre = time.time()
    for train, val in kf_pre.split(labels_pre):
        # Subsets of training data and validation data after cross validation split
        X_train, y_train, X_val, y_val = inputs_pre[train], labels_pre[train], inputs_pre[val], labels_pre[val]
        subset_num_pre = len(y_val)
        #add the number of the validation data in this folder
        subset_nums_pre.append(subset_num_pre)
        
        # Loss and Optimer
        mlp_pre = MLP(input_size, hidden_size, output_size, drop_rate=dr_score)
        criterion = nn.MSELoss()
        # Adam Regularization
        optimizer_pre = torch.optim.Adam(mlp_pre.parameters(), lr=learning_rate, weight_decay=0.01)
        scheduler = ExponentialLR(optimizer_pre, gamma = lr_exponential)
        MLP_learning_curve_pre = []       
        epoch_learning_rate_pre = []
        for epoch in range(num_epochs):

            optimizer_pre.zero_grad()
            # torch to variable
            X_train_variable = Variable(torch.from_numpy(X_train).float())
            y_train_variable = Variable(torch.from_numpy(y_train).float())

            X_val_variable = Variable(torch.from_numpy(X_val).float())
            y_val_variable = Variable(torch.from_numpy(y_val).float())

            # train model
            mlp_pre.train()
            outputs = mlp_pre(X_train_variable)
            loss = criterion(outputs, y_train_variable)
            loss.backward()
            optimizer_pre.step()
            scheduler.step()
            
            for param_group in optimizer_pre.param_groups:
                epoch_learning_rate_pre += [ param_group['lr'] ]
            """
            if epoch % 100 == 0:
                print('Epoch [%d/%d], Validation Loss: %.4f'
                      % (epoch + 1, num_epochs, loss.data[0]))
            """
            MLP_learning_curve_pre.append(loss.data[0])
        
        #print(epoch_learning_rate)
        epoch_learning_rates_pre.append(epoch_learning_rate_pre)
        KFolder_set_pre.append(len(val))
        # validate model
        mlp_pre.eval()  # for the case of dropout-layer and Batch-normalization
        outputs = mlp_pre(X_val_variable)
        validation_loss = criterion(outputs, y_val_variable)
        
        print(validation_loss.data[0])
        
        MLP_learning_curves_pre.append(MLP_learning_curve_pre)
        #print(validation_loss.data[0])
        print("End of one cross validation subset")
    
    #calculate average validation error
    end_time_pre = time.time() - start_time_pre
    print('Runtime of training: %.4fs' %end_time_pre)
    
    #print(KFolder_set) # 89, 88, 88
    ###
    #print(MLP_learning_avg_curves)
    
    for i in range(len(MLP_learning_curves_pre)):
        for j in range(len(MLP_learning_curves_pre[i])):
            MLP_learning_avg_curve_pre[j] += MLP_learning_curves_pre[i][j] * KFolder_set_pre[i] / len(labels_pre)
            
                
    ### 
    print("Final Error after 40 epochs: %.4f " % MLP_learning_avg_curve_pre[-1])
    #print(MLP_learning_avg_curve_pre[-1])
       
    t_idx = np.arange(1, num_epochs+1)
    
    # all sub curves in 3-folder cross validation 
    [plt.plot(t_idx, lc) for lc in MLP_learning_curves_pre]
    plt.title("Subset of learning curves")
    plt.xlabel("Number of epochs")
    plt.ylabel("Validation error")
    plt.show()
    
    
    [plt.plot(t_idx, MLP_learning_avg_curve_pre)]
    plt.title("cross validation learning curve")
    plt.xlabel("Number of epochs")
    plt.ylabel("Validation error")
    plt.show()

    """
    [plt.plot(t_idx, lr) for lr in epoch_learning_rates_pre]
    plt.title("Learning rates")
    plt.xlabel("Number of epochs")
    plt.ylabel("Learning rate")
    plt.show()
    """
    
    #Histogram and CDF over the final error
    sorted_MLP_learning_curve_pre = np.sort(MLP_learning_avg_curve_pre)
    h = plt.hist(sorted_MLP_learning_curve_pre, bins=20)
    plt.title("Histogram over the final error")
    plt.show()
    
    yvals = np.arange(len(sorted_MLP_learning_curve_pre))/float(len(sorted_MLP_learning_curve_pre))
    plt.plot(sorted_MLP_learning_curve_pre, yvals)
    plt.title("Empirical CDF")
    plt.xlabel("y(x, t=40)")
    plt.ylabel("CDF(y)")
    plt.show()
    
    #store learning curves into json file but not used in this project
    """
    json:
        {
        "final_performance":  MLP_learning_avg_curve[-1],
        
        "runtime" : end_time,
        
        "config" : {"batch_size": 509.70500417364815, 
        "log2_n_units_2": 8.276062946870212, 
        "log10_learning_rate": -2.41971978428842, 
        "log2_n_units_3": 9.674918930141073, 
        "log2_n_units_1": 4.978802069501831}, 
        
        "learning_curve": MLP_learning_avg_curve    
        }
    """
    
    # store traing results with raw data
    json_string_raw = '{"final_performance": ' + str(MLP_learning_avg_curve_raw[-1]) +', '
    json_string_raw += '"rumtime": ' +str(end_time_raw) + ', '
    json_string_raw += '"config": {' 
    json_string_raw += '"drop_rate": ' + str(dr_score) + ', '
    json_string_raw += '"learning_rate": ' + str(learning_rate) + ', '
    json_string_raw += '"learning_rate": ' + str(epoch_learning_rates_pre) + ', '
    json_string_raw += '"num_epochs": ' + str(num_epochs)
    json_string_raw += '}, '
    json_string_raw += '"config_id": ' + str(CONFIG_ID) + ', '
    json_string_raw += '"learning_curve": ' + str(MLP_learning_avg_curve_raw)
    json_string_raw += '}'
    
    
    #print(json_string)
    parsed_json_raw = json.loads(json_string_raw)
    
    #print(parsed_json['config'])
    #configs_json = parsed_json['config']
    #print(configs_json['drop_rate'])
    
    
    with open('./data/results/mlp/config_raw_'+str(CONFIG_ID)+'.json', 'w') as outfile:
        json.dump(parsed_json_raw, outfile)
    
    
    # store traing results with preprocessed data
    json_string_pre = '{"final_performance": ' + str(MLP_learning_avg_curve_pre[-1]) +', '
    json_string_pre += '"rumtime": ' +str(end_time_pre) + ', '
    json_string_pre += '"config": {' 
    json_string_pre += '"drop_rate": ' + str(dr_score) + ', '
    json_string_pre += '"learning_rate": ' + str(learning_rate) + ', '
    json_string_pre += '"learning_rate": ' + str(epoch_learning_rates_pre) + ', '
    json_string_pre += '"num_epochs": ' + str(num_epochs)
    json_string_pre += '}, '
    json_string_pre += '"config_id": ' + str(CONFIG_ID) + ', '
    json_string_pre += '"learning_curve": ' + str(MLP_learning_avg_curve_pre)
    json_string_pre += '}'
    
    
    #print(json_string)
    parsed_json_pre = json.loads(json_string_pre)
    
    #print(parsed_json['config'])
    #configs_json = parsed_json['config']
    #print(configs_json['drop_rate'])
    
    
    with open('./data/results/mlp/config_pre_'+str(CONFIG_ID)+'.json', 'w') as outfile:
        json.dump(parsed_json_pre, outfile)
    
    
    
    