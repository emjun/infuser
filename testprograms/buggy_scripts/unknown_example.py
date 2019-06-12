"""
buggy lines: 24, 27
"""
from scipy.stats import ttest_ind, mannwhitneyu
import pandas as pd

"""
https://github.com/scidam/pyworks/blob/4bc0cff3887e8face5ab443113ee56a7b6222a2c/res26_NOV_2018/analyz.py
WORKS FINE WITH INFUSER
"""

"WRANGLING"
data = pd.read_csv('../data/out.csv', header=None)

x = data.loc[:,0].astype(float).values
y = data.loc[~pd.isnull(data.loc[:,1]),1].astype(float).values

"ANALYSIS"

""" ################### README ###################################
debatable wheter is buggy or not - we just don't have the information
"""
print("Column # 1 significantly differs from #2 (t-test)")
t_test = ttest_ind(x,y) 

print("Column # 1 significantly differs from #2 (Mann-Whitney test)")
mann = mannwhitneyu(x,y)