from scipy.stats import mannwhitneyu, median_test, rv_discrete
import csv
import numpy as np
import collections 
from functools import reduce
import random

"""
https://github.com/pierreTklein/cogs_444_data_analysis/blob/6aeb2e37afdc1012977bb69aceb0f66e6886fc61/analyze.py
WORKS FINE, HAD TO REWRITE: LIST COMPREHENSION HAD TO BE REMOVED
GIVES OUT WARNINGS THAT SHOULD NOT BE GIVEN OUT: WE SHOULDN'T GIVE ANY, BUT 
GIVE ONE FOR EVERY TEST CONDUCTED
"""

"WRANGLING"

filename_week_1 = '../data/neg_correlation_week_1.csv'
filename_week_2 = '../data/neg_correlation_week_2.csv'
          
week_1 = pd.read_csv(filename_week_1)
week_2 = pd.read_csv(filename_week_2)




"""
# INFUSER DOES NOT LIKE THIS LIST COMPREHENSION SO WE DO IT MANUALLY BELOW
w2_cr, w1_di, w1_comp, w1_comf, w1_nh = [
    [row[i] for row in week_1] for i in range(len(week_1[0]))]

w2_cr, w2_di, w2_comp, w2_comf, w2_nh = [
    [row[i] for row in week_2] for i in range(len(week_2[0]))]

"""

w1_cr = week_1['crowdSize']
w1_di = week_1['difficulty']
w1_comp = week_1['sessionsCompleted']
w1_comf = week_1['comfort']
w1_nh = week_1['numHelped']

w2_cr = week_2['crowdSize']
w2_di = week_2['difficulty']
w2_comp = week_2['sessionsCompleted']
w2_comf = week_2['comfort']
w2_nh = week_2['numHelped']




"ANALYSIS"


'''
Run Mann Whitney U test on all 5 variables across week 1 and week 2. 
week_1, week_2 is of the same format as the output of genData.
Return the results for all 5 variables.
'''

nh_mwu = mannwhitneyu(w1_nh, w2_nh)
cr_mwu = mannwhitneyu(w1_cr, w2_cr)
comp_mwu = mannwhitneyu(w1_comp, w2_comp)
di_mwu = mannwhitneyu(w1_di, w2_di)
comf_mwu = mannwhitneyu(w1_comf, w2_comf)
table = [
    ('Question', 'U Statistic', 'P-Value'),
    ('Number Helped', nh_mwu[0], nh_mwu[1]),
    ('Crowd Size', cr_mwu[0], cr_mwu[1]),
    ('Completed', comp_mwu[0], comp_mwu[1]),
    ('Difficulty', di_mwu[0], di_mwu[1]),
    ('Comfort', comf_mwu[0], comf_mwu[1])
]
print("Matt Whitney U Test Results", table)
