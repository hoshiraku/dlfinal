#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  6 14:54:21 2018

@author: hoshiraku
"""

"""
configuration:

batch_size
log2_n_units_2
log10_learning_rate
log2_n_units_3
log2_n_units_1 


values at different time:
    value_t1
    value_t2
    value_t3
    value_t4 

use these parameters to predict:
    value_t5

etc.....


configuration (5 parameters)
values at different time:
    value_t36
    value_t37
    value_t38
    value_t39
use these parameters to predict:
    value_t40


36 group data set in 1 configuration

there are 265 groups of configurations

there are 265 * 36 = 9540 sets of data
"""

# X: 9540 * 9
# y: 9540 * 1



#set a model to predict
from sklearn import linear_model

 

# filt a model
#lm = linear_model.LinearRegression()
# Ridge Regression
lm = linear_model.Ridge(alpha = 0.5)




