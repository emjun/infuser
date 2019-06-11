'''
buggy lines: 29
'''
import os 
import numpy as np
import pandas as pd
from scipy import stats


"WRANGLING"

df = pd.read_csv('../data/cars.csv')


sticker_price_diff = df['2010_Price'] - df['2019_Price']

actual_price_2010 = df['2010_Price'] - df['2010_PriceDiscount'] * df['2010_Price']
actual_price_2019 = df['2019_Price'] - df['2019_PriceDiscount'] * df['2019_Price']

print(actual_price_2019)

"ANALYSIS"

# these two should work out
stats.mannwhitneyu(df['2010_Price'], df['2019_Price'])
stats.mannwhitneyu(actual_price_2010, actual_price_2019)



