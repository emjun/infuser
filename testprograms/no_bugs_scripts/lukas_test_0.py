# -*- coding: utf-8 -*-
"""
Created on Tue May  7 15:52:47 2019

@author: Lukas Blass
"""

import os 
import numpy as np
import pandas as pd
from scipy import stats

"WRANGLING"
'''
# gotta know this command or look it up
base_path = '/Users/emjun/Git/infuser/testprograms/data'
file_name = 'data_0.csv'
file_path = os.path.join(base_path, file_name)
data = pd.read_csv(file_path)
'''
data = pd.read_csv('../data/data_0.csv')

#import pdb; pdb.set_trace()

cond_a = data[data['condition'] == 'a']
cond_b = data[data['condition'] == 'b']

"ANALYSIS"

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

stat, p_val = stats.ttest_ind(cond_a['time'], cond_b['time'])
print('p_val time: ' + str(p_val))
if p_val < 0.1:
    print('unequal means for time')
else:
    print('equal means for time')


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

stat, p_val = stats.ttest_ind(cond_a['accuracy'], cond_b['accuracy'], equal_var=True)
print('p_val = ' + str(p_val))

if p_val < 0.1:
    print('unequal means for accuracy')
else:
    print('equal means for accuracy')
