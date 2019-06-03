# -*- coding: utf-8 -*-
'''
buggy lines: 58
'''

"""
Created on Tue May  7 16:25:46 2019

@author: Lukas Blass
"""

import numpy as np
import pandas as pd
from scipy import stats

"WRANGLING"
data = pd.read_csv('../data/timing.csv')

naive = data['naive']
cache = data['caching']
fork = data['forking']

avg = (naive + cache + fork) / 3

df =  pd.concat([avg, data['run']], axis=1)

naive_run1 = naive[df['run'] == 1]
naive_run2 = naive[df['run'] == 2]

fork_run1 = fork[df['run'] == 1]
fork_run2 = fork[df['run'] == 2]

fork_run_1_2 = fork_run1.append(fork_run2, ignore_index=True)
naive_run_1_2 = naive_run1.append(naive_run2, ignore_index=True)

# want to have a 'manual' look at data first
diff = fork_run_1_2 - naive_run_1_2
print(diff)


"ANALYSIS"

# now this should be allowed
test_result = stats.ttest_ind(fork_run_1_2, naive_run_1_2)
print(test_result)

# let's do something random
fork_runs = pd.concat([fork, data['run']], axis=1)
naive_runs = pd.concat([naive, data['run']], axis=1)

# this should again work since we essentially use the same column twice
test_result = stats.mannwhitneyu(fork_runs['run'], naive_runs['run'])

# now let's still be random, but also buggy
long_fork_run = fork_runs['forking'].append(fork_runs['run'])
long_naive_run = naive_runs['naive'].append(naive_runs['run'])

test_result = stats.mannwhitneyu(long_fork_run, long_naive_run)
print(test_result)

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
