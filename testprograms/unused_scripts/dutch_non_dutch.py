import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt 
import seaborn as sns
import scipy.stats as stats
from pandas import DataFrame 
import math 
from scipy.stats import mannwhitneyu

"""
https://github.com/brunnatorino/dutch-nondutch-stats/blob/3367c6ee43a5826cec7b5526d2a84550294b9837/nationality.py
WORKS FINE WITH INUFSER
"""

"WRANGLING"
## importing excel file with data and calling columns for better data handling
## culture refers to region in the world person is from, will not use it specifically for this code 
##since this is just for visualization purposes

df_initial = pd.read_excel('../data/thewhyprobs.xlsx', names=['time','ID','region','culture','perc_P_50c','perc_R_50c',
                                            'perc_P_20','perc_R_20'])
## deleting columns 

del df_initial['time']
del df_initial['ID']
del df_initial['culture']

## deleting values outside of possible ranges 

indexNames = df_initial[((df_initial['perc_P_50c'] > 50) | (df_initial['perc_P_50c'] < 0))
                        & ((df_initial['perc_R_50c'] < 0) | (df_initial['perc_R_50c'] > 50)) 
                        & ((df_initial['perc_P_20'] < 0) | (df_initial['perc_P_20'] > 20))
                        & ((df_initial['perc_R_20'] < 0) | (df_initial['perc_R_20'] > 20))].index

## returns which ID is invalid according to possible range

## deletes values outside range of the dataframe 

df = df_initial.drop(indexNames)


## setting up for dutch vs. non dutch comparison 

df_dutch = df[(df['region'] == 'Dutch')]


df_nondutch = df[(df['region'] != 'Dutch')]

print(df_dutch)
print(df_nondutch)

print(df_dutch.count())
print(df_nondutch.count())

##counts how many inputs in each 

## no need to print these because it's just formatting data for analysis

## EVERYTIME you run this, run the first kernel/data calling FIRST or else you will divide again 

df_dutch[['perc_P_50c','perc_R_50c']] = df_dutch[['perc_P_50c','perc_R_50c']]/50
df_dutch[['perc_P_20','perc_R_20']] = df_dutch[['perc_P_20','perc_R_20']]/20

df_nondutch[['perc_P_20','perc_R_20']] = df_nondutch[['perc_P_20','perc_R_20']]/20
df_nondutch[['perc_P_50c','perc_R_50c']] = df_nondutch[['perc_P_50c','perc_R_50c']]/50

print (df_dutch)
print (df_nondutch)


"ANALYSIS"


## PROPOSER, PAYOFF = 20, DUTCH VS. NON DUTCH


print('Test Dutch vs. Non-Dutch for Proposers when Payoff is 20')
stat, p = mannwhitneyu(df_dutch['perc_P_20'], df_nondutch['perc_P_20'])
print('Statistics=%.3f, p=%.3f' % (stat, p))
alpha = 0.05
if p > alpha:
    print('Same distribution (fail to reject H0)')
else:
    print('Different distribution (reject H0)')

## distribution plots for class presentation

sns_1 = sns.distplot(df_dutch['perc_P_20'], hist=False, rug=True);  ##blue
sns_2 = sns.distplot(df_nondutch['perc_P_20'], hist=False, rug=True);  ##orange


print("               ")
print("median of dutch proposer 20 is:")
print(df_dutch['perc_P_20'].median())
print("median of non-dutch proposer 20 is:")
print(df_nondutch['perc_P_20'].median())

print("               ")
print("mean of dutch proposer 20 is:")
print(df_dutch['perc_P_20'].mean())
print("mean of non-dutch proposer 20 is:")
print(df_nondutch['perc_P_20'].mean())



## RESPONDER, PAYOFF = 20, DUTCH VS. NON DUTCH

print('')
print('Test Dutch vs. Non-Dutch for Responders when Payoff is 20')
stat, p = mannwhitneyu(df_dutch['perc_R_20'], df_nondutch['perc_R_20'])
print('Statistics=%.3f, p=%.3f' % (stat, p))
alpha = 0.05
if p > alpha:
    print('Same distribution (fail to reject H0)')
else:
    print('Different distribution (reject H0)')

sns_5 = sns.distplot(df_dutch['perc_R_20'], hist=False, rug=True);  ##blue
sns_6 = sns.distplot(df_nondutch['perc_R_20'], hist=False, rug=True);  ##orange


print("               ")
print("median of dutch responder 20 is:")
print(df_dutch['perc_R_20'].median())
print("median of non-dutch responder 20 is:")
print(df_nondutch['perc_R_20'].median())

print("               ")
print("mean of dutch responder 20 is:")
print(df_dutch['perc_R_20'].mean())
print("mean of non-dutch responder 20 is:")
print(df_nondutch['perc_R_20'].mean())



## PROPOSER, PAYOFF = 50 cents, DUTCH VS. NON DUTCH

print('Test Dutch vs. Non-Dutch for Proposers when Payoff is 50c')
stat, p = mannwhitneyu(df_dutch['perc_P_50c'], df_nondutch['perc_P_50c'])
print('Statistics=%.3f, p=%.3f' % (stat, p))
alpha = 0.05
if p > alpha:
    print('Same distribution (fail to reject H0)')
else:
    print('Different distribution (reject H0)')

sns_3 = sns.distplot(df_dutch['perc_P_50c'], hist=False, rug=True);  ##blue
sns_4 = sns.distplot(df_nondutch['perc_P_50c'], hist=False, rug=True);  ##orange

print("               ")
print("median of dutch proposer 20 is:")
print(df_dutch['perc_P_50c'].median())
print("median of non-dutch proposer 20 is:")
print(df_nondutch['perc_P_50c'].median())

print("               ")
print("mean of dutch proposer 20 is:")
print(df_dutch['perc_P_50c'].mean())
print("mean of non-dutch proposer 20 is:")
print(df_nondutch['perc_P_50c'].mean())



## RESPONDER, PAYOFF = 50c, DUTCH VS. NON DUTCH

print('Test Dutch vs. Non-Dutch for Responders when Payoff is 50c')
stat, p = mannwhitneyu(df_dutch['perc_R_50c'], df_nondutch['perc_R_50c'])
print('Statistics=%.3f, p=%.3f' % (stat, p))
alpha = 0.05
if p > alpha:
    print('Same distribution (fail to reject H0)')
else:
    print('Different distribution (reject H0)')

sns_7 = sns.distplot(df_dutch['perc_R_50c'], hist=False, rug=True);  ##blue
sns_8 = sns.distplot(df_nondutch['perc_R_50c'], hist=False, rug=True);  ##orange

print("               ")
print("median of dutch responder 20 is:")
print(df_dutch['perc_R_20'].median())
print("median of non-dutch responder 20 is:")
print(df_nondutch['perc_R_20'].median())

print("               ")
print("mean of dutch responder 20 is:")
print(df_dutch['perc_R_20'].mean())
print("mean of non-dutch responder 20 is:")
print(df_nondutch['perc_R_20'].mean())





















"""
import plotly.offline
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)

import pandas as pd
import numpy as np
import plotly.offline as py
import plotly.graph_objs as go
import json

## rounding numbers to 4 decimals (only necessary with means and p-value)

def truncate(n, decimals=4):
    multiplier = 10 ** decimals
    return int(n * multiplier) / multiplier

## create a table with the stats, p-value and means/medians for better visualization 
## will export tables and graphs to google slides (through API) as well

med_20R_Dutch = df_dutch['perc_R_20'].median()
med_20R_NonDutch = df_nondutch['perc_R_20'].median()

mean_20R_Dutch = truncate(df_dutch['perc_R_20'].mean())
mean_20R_NonDutch = truncate(df_nondutch['perc_R_20'].mean())

med_20P_Dutch = df_dutch['perc_P_20'].median()
med_20P_NonDutch = df_nondutch['perc_P_20'].median()

mean_20P_NonDutch = truncate(df_nondutch['perc_P_20'].mean())
mean_20P_Dutch = truncate(df_dutch['perc_P_20'].mean())

p_value = truncate(p)

## styled table

trace1 = go.Table(
    header=dict(values=['','Mean','Median','Mann Whitney Statistic','P-value'],
                line = dict(color='#8c564b'),
                fill = dict(color='#17becf'),
                align = ['left'] * 5),
    cells=dict(values=[['Proposer 20 euros Dutch', 'Proposer 20 euros Non-Dutch','Responder 20 euros Dutch',
                       'Responder 20 euros Non-Dutch'],
                       [mean_20P_Dutch,mean_20P_NonDutch,mean_20R_Dutch, mean_20R_NonDutch],
                      [med_20R_Dutch,med_20R_NonDutch,med_20R_Dutch, med_20R_NonDutch], [stat],[p_value]],
               line = dict(color='#8c564b'),
               fill = dict(color='#EDFAFF'),
               align = ['left'] * 5))

layout = dict(width=700, height=500)
data = [trace1]
fig = dict(data=data, layout=layout)
py.iplot(fig, filename = 'styled_table_P')
"""