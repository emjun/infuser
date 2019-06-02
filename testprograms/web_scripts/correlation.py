# Importing required libraries
import pandas as pd
import researchpy as rp
from scipy import stats

# WRANGLING
df = pd.read_csv("https://raw.githubusercontent.com/Opensourcefordatascience/Data-sets/master/diamonds.csv")


# ANALYSIS

stats.levene(df['carat'], df['price'])

df['carat'].corr(df['price'])

df['carat'].corr(df['price'], method= 'spearman')

stats.pearsonr(df['carat'], df['price'])
stats.spearmanr(df['carat'], df['price'])

stats.kendalltau(df['carat'], df['price'])

corr_type, corr_matrix, corr_ps = rp.corr_case(df[['carat', 'price', 'depth']])

rp.corr_pair(df[['carat', 'price', 'depth']])