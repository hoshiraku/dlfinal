#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 20 03:32:04 2018

@author: hoshiraku
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 11 21:55:40 2018

@author: hoshiraku
"""

import numpy as np
from data import load_data_learning_curve

from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt

N_epochs = 10 # change different N epochs: 5, 10, 20, random

if __name__ == '__main__':
    # many baselines
    
    # Random Forest Regression
    
    # load X, y
    inputs, labels = load_data_learning_curve(N_epochs)
    #print(inputs)
    #labels = labels.ravel()
    #print(labels)
    #reg_ranfor = RandomForestRegressor(max_depth=2, random_state=0)
    #reg_ranfor.fit(inputs, labels)
    
    # cross validation for MLP model
    
    kf = KFold(n_splits=3, shuffle=True)
    
    print("Random Forest model:")
    avg_mse_ranfor = 0.0
    avg_score_ranfor = 0.0
    
    for train, val in kf.split(labels):
        X_train, y_train, X_val, y_val = inputs[train], labels[train], inputs[val], labels[val]
        #construct a random forest regression model
        reg_ranfor = RandomForestRegressor(max_depth=3, random_state=0)
        #convert column vector to 1-d array
        y_train = y_train.ravel()
        y_val = y_val.ravel()
        #train the model
        reg_ranfor.fit(X_train, y_train)
        y_pred = reg_ranfor.predict(X_val)
        #print(y_pred)
        score_ranfor = reg_ranfor.score(X_val, y_val)
        mse_ranfor = mean_squared_error(y_val, y_pred)
        
        avg_mse_ranfor += mse_ranfor * len(y_val) / len(labels)
        avg_score_ranfor += score_ranfor * len(y_val) / len(labels)
        print("one cross validation MSE: % f" % mse_ranfor)
        print("one cross validation score: % f" % score_ranfor)
        
    
    print("3-fold cross validation MSE: % f" %avg_mse_ranfor)
    print("3-fold cross validation score: % f" %avg_score_ranfor)
    
    # Decision Tree Regression
    print("Decision Tree model:")
    avg_mse_detr = 0.0
    avg_score_detr = 0.0
    
    for train, val in kf.split(labels):
        X_train, y_train, X_val, y_val = inputs[train], labels[train], inputs[val], labels[val]
        #construct a decision tress regression model
        reg_detr = DecisionTreeRegressor(max_depth=2)
        #train the model
        reg_detr.fit(X_train, y_train)
        y_pred = reg_detr.predict(X_val)
        score_detr = reg_detr.score(X_val, y_val)
        mse_detr = mean_squared_error(y_val, y_pred)
        
        avg_mse_detr += mse_detr * len(y_val) / len(labels)
        avg_score_detr += score_detr * len(y_val) / len(labels)
        
        print("one cross validation MSE: % f" % mse_detr)
        print("one cross validation score: % f" % score_detr)
    
    print("3-fold cross validation MSE: % f" %avg_mse_detr)
    print("3-fold cross validation score: % f" %avg_score_detr)
    
    # Linear Regression
    print("Linear model:")
    avg_mse_lin = 0.0
    avg_score_lin = 0.0
    
    for train, val in kf.split(labels):
        X_train, y_train, X_val, y_val = inputs[train], labels[train], inputs[val], labels[val]
        #construct a decision tress regression model
        reg_lin = LinearRegression()
        #train the model
        reg_lin.fit(X_train, y_train)
        y_pred = reg_lin.predict(X_val)
        score_lin = reg_lin.score(X_val, y_val)
        mse_lin = mean_squared_error(y_val, y_pred)
        
        avg_mse_lin += mse_lin * len(y_val) / len(labels)
        avg_score_lin += score_lin * len(y_val) / len(labels)
        
        print("one cross validation MSE: % f" % mse_lin)
        print("one cross validation score: % f" % score_lin)
    
    print("3-fold cross validation MSE: % f" %avg_mse_lin)
    print("3-fold cross validation score: % f" %avg_score_lin)
    