from scipy import stats
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
#import statsmodels.api as sm
"""
https://github.com/mtinti/jennie_analysis/blob/5cfe47fc9e5895fb6434538f3c921793726f61b2/utility/infectivity.py
doesn't work with infuser, throw keyError on line 19
"""


"WRANGLING"
df = pd.DataFrame.from_csv('anna_data/infectivity_dataset.txt',sep='\t')
df['counts'] = df['count']#*10+1

#df['counts'] = np.log10(df['counts'] )
del df['count']


df['condition'] = ['minus_dox' if n == 'minus_Dox' else n for n in df['condition']]


x=df[df['condition']=='WT']['count'].values
y=df[df['condition']=='dKO4']['count'].values



"ANALYSIS"
res=stats.mannwhitneyu(x=x,y=y)
x1, x2 = 0, 1
y, h, col = x.max() + 0.1, 0.1, 'k'
x=df[df['condition']=='WT']['count'].values
y=df[df['condition']=='dKO2']['count'].values
res=stats.mannwhitneyu(x=x,y=y)
x1, x2 = 0, 2
y, h, col = x.max() + 0.5, 0.1, 'k'
x=df[df['condition']=='WT']['count'].values
y=df[df['condition']=='dKO3']['count'].values
res=stats.mannwhitneyu(x=x,y=y)

x1, x2 = 0, 3
y, h, col = x.max() + 1, 0.1, 'k'

x=df[df['condition']=='plus_dox']['count'].values
y=df[df['condition']=='minus_dox']['count'].values
res=stats.mannwhitneyu(x=x,y=y)
