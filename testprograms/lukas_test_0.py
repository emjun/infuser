# -*- coding: utf-8 -*-
"""
Created on Tue May  7 15:52:47 2019

@author: Lukas Blass
"""

import pandas as pd
import numpy as np
from scipy import stats
from infuser import infuser


# WRANGLING

# gotta know this command or look it up
infuser.wrangling()
data = pd.read_csv('data/data_0.csv')

cond_a = data[data['condition'] == 'a']
cond_b = data[data['condition'] == 'b']

# ANALYSIS
infuser.analyzing()
# first do some hand calculations
print('Mean, Median, Sigma for a:')
print(np.mean(cond_a['time']))
print(np.median(cond_a['time']))
print(np.sqrt(np.var(cond_a['time'])))

print('Mean, Median, Sigma for b:')
print(np.mean(cond_b['time']))
print(np.median(cond_b['time']))
print(np.sqrt(np.var(cond_b['time'])))

# then let the t-test decide
# do we have equal variance?
b = abs(np.sqrt(np.var(cond_a['time'])) - np.sqrt(np.var(cond_b['time']))) < 10

# stat, p_val = stats.ttest_ind(cond_a['time'], cond_b['time'], equal_var=b)
# print('p_val time: ' + str(p_val))
# if p_val < 0.1:
#     print('unequal means for time')
# else:
#     print('equal means for time')


print('=========================')
# now do the same for the accuracy property

print('Mean, Median, Sigma for a:')
print(np.mean(cond_a['accuracy']))
print(np.median(cond_a['accuracy']))
print(np.sqrt(np.var(cond_a['accuracy'])))

print('Mean, Median, Sigma for b:')
print(np.mean(cond_b['accuracy']))
print(np.median(cond_b['accuracy']))
print(np.sqrt(np.var(cond_b['accuracy'])))


b = abs(np.sqrt(np.var(cond_a['accuracy'])) - np.sqrt(np.var(cond_b['accuracy']))) < 10

# stat, p_val = stats.ttest_ind(cond_a['accuracy'], cond_b['accuracy'], equal_var=True)
# print('p_val = ' + str(p_val))

# if p_val < 0.1:
#     print('unequal means for accuracy')
# else:
#     print('equal means for accuracy')
