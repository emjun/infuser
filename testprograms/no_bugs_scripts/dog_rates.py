import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re
from scipy import stats
"""
https://github.com/apaunescu/DS/blob/fbecc6c9e73aaf4519bb52c3886f411bc9de2b35/e7/dog-rates.ipynb
WORKS PERFERCTLY FINE
"""
"WRANGLING"

df = pd.read_csv("../data/dog_rates_tweets.csv", parse_dates = ['created_at'])

def find_rating(text):
    match = re.search(r'(\d+(\.\d+)?)/10', text)
    if match:
        rating = float(match.group(1))
        if (rating > 25):
            return None      
        return rating
    return None

df['rating'] = df['text'].apply(find_rating)

def timestamp(datetime):
    return datetime.timestamp()

df['timestamp'] = df['created_at'].apply(timestamp)

df.dropna(inplace=True)


"ANALYSIS"

myLinregress = stats.linregress(df['timestamp'], df['rating'])

plt.xticks(rotation=25)
plt.plot(df['created_at'], df['timestamp']*myLinregress.slope + myLinregress.intercept, 'r-', alpha=1.0)
plt.plot(df['created_at'], df['rating'], 'b.', linewidth=3)
plt.show()

print(myLinregress.pvalue)

residuals = df['rating'] - (myLinregress.slope*df['timestamp'] + myLinregress.intercept)