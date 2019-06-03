# Importing required libraries
import pandas as pd
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
