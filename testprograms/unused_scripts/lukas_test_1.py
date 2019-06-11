# -*- coding: utf-8 -*-
"""
Created on Tue May  7 16:25:46 2019

@author: Lukas Blass
"""

import os
import numpy as np
import pandas as pd

"WRANGLING"
base_path = '/Users/emjun/Git/infuser/testprograms/data'
file_name = 'timing.csv'
file_path = os.path.join(base_path, file_name)

#file_path = '../data/timing.csv'
data = pd.read_csv(file_path)
print(data)

"ANALYSIS"
subjects = ['tax', 'tictactoe', 'triangle']
approaches = ['forking'] #, 'caching', 'forking']

# comparing the different runs within a  strategy
for a in approaches: # for a single strategy    
    for s in subjects: # cover all subjects
        for run in range(1,6): # over all different runs
            # import pdb; pdb.set_trace()
            r = data[(data['run'] == run) & (data['subject'] == s)]
            print(a + ' ' + s + ' ' + str(run) + ' ' + str(np.mean(r[a])))
            
# comparing different strategies
approaches = ['forking', 'caching', 'naive']            

fork = data['forking']
cache = data['caching']
naive = data['naive']

for s in subjects:
    fork_subj = fork[data['subject'] == s]
    cache_subj = cache[data['subject'] == s]
    naive_subj = naive[data['subject'] == s]
    
    fork_mean = np.mean(fork_subj)
    cache_mean = np.mean(cache_subj)
    naive_mean = np.mean(naive_subj)
    
    print('fork_mean: ' + str(fork_mean))
    print('cache_mean: ' + str(cache_mean))
    print('naive_mean: ' + str(naive_mean))

     
        
        
        
        
        
        
        
        
        
        
        
