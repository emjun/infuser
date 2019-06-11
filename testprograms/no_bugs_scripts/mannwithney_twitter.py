
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import mannwhitneyu
import math
#https://github.com/zhongshj/WSE_Streaming/blob/472bd1c479e95141c98b47f5300012a5accdd4c3/predata.py

"WRANGLING"

df = pd.read_csv("../data/task2_data.csv")

#4 complete datasets
nr_entities = df['#entities']
nr_entityTypes = df['#entityTypes']
nr_tweetsPosted = df['#tweetsPosted']
nr_sentiment = df['sentiment']
#entities: 0-11, entityTypes: 0-4, tweetsPosted: 0-130000, sentiment: 0,1

#separate each dataset to r=relevant, ir=irrelevant
r_e = df[df['relevanceJudge']==1]['#entities']
ir_e = df[df['relevanceJudge']==0]['#entities']
r_eT = df[df['relevanceJudge']==1]['#entityTypes']
ir_eT = df[df['relevanceJudge']==0]['#entityTypes']
r_tP = df[df['relevanceJudge']==1]['#tweetsPosted']
ir_tP = df[df['relevanceJudge']==0]['#tweetsPosted']
r_s = df[df['relevanceJudge']==1]['sentiment']
ir_s = df[df['relevanceJudge']==0]['sentiment']


"ANALYSIS"
#Do Mann-Whitney U test and get p-value
u,pvalue_e = mannwhitneyu(r_e,ir_e)
u,pvalue_eT = mannwhitneyu(r_eT,ir_eT)
u,pvalue_tP = mannwhitneyu(r_tP,ir_tP)
u,pvalue_s = mannwhitneyu(r_s,ir_s)
print (pvalue_s)

#print pvalue_eT
#r_s.plot(kind='hist',title='Relevant')
#plt.show()
#ir_s.plot(kind='hist',title='Non-relevant')
#plt.show()
#r_tP.plot(kind='box',logy=True,label='Relevant')
#plt.show()
#ir_tP.plot(kind='box',logy=True,label='Non-relevant')
#plt.show()
#boxplot

#fig = plt.figure(1,figsize=(9,6))
#ax = fig.add_subplot(111)
#ax.set_xticklabels(['Relevant', 'Non-relevant'])
#bp = ax.boxplot([r_tP,ir_tP],labels=('Relevant','Non-relevant'))
#ax.set_yscale("log")
#plt.show()
