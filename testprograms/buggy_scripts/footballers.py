"""
buggy lines: 74
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import mannwhitneyu, shapiro, norm

"""
https://github.com/kontrybutor/tsbd_sem2/blob/7e97b92900d092c2eb67a17167ae9f9df4ae0d0d/JPwAD/ex1/ex3.py
WORKS FINE WITH INFUSER
"""


"WRANGLING"

footballers_filename = '../data/footballers.csv'


def load_dataset(filename):
    df = pd.read_csv(filename, sep=";")

    return df


def plot_density_function(data1, data2):
    plt.figure(1, figsize=(12, 10))
    sns.distplot(data1, hist=False, kde=True,
                 kde_kws={'shade': True, 'linewidth': 3},
                 label='starting_pitchers')
    sns.distplot(data2, hist=False, kde=True,
                 kde_kws={'shade': True, 'linewidth': 3},
                 label='relief_pitchers')
    plt.title('Histogram')
    plt.xlabel('Weight(pounds)')
    plt.ylabel('Amount')
    plt.grid(True)
    plt.show()


def plot_histogram(data1, data2):
    plt.figure()
    plt.subplot(211)
    mu_1, std_1 = norm.fit(data1)
    plt.hist(data1, bins=30, density=True)
    x = np.linspace(data1.min(), data1.max(), 1000)
    y = norm.pdf(x, mu_1, std_1)
    plt.plot(x, y)
    plt.title('Starting Pitchers')
    plt.legend(('Gauss approximation', 'Weight'))
    plt.ylabel('Weight')
    plt.grid(True)
    plt.subplot(212)
    mu_2, std_2 = norm.fit(data2)
    plt.hist(data2, bins=30, density=True)
    x = np.linspace(data2.min(), data2.max(), 1000)
    y = norm.pdf(x, mu_2, std_2)
    plt.plot(x, y)
    plt.legend(('Gauss approximation', 'Weight'))
    plt.title('Relief Pitchers')
    plt.ylabel('Weight')
    plt.grid(True)
    plt.show()
    

print('-----------------------------------')
footballers = load_dataset(footballers_filename)
starting_pitchers = footballers[footballers['Position'] == 'Starting_Pitcher']['Weight(pounds)']

#relief_pitchers = footballers[footballers['Position'] == 'Relief_Pitcher']['Weight(pounds)']

"""extract height instead of weight ==> should give error"""
relief_pitchers = footballers[footballers['Position'] == 'Relief_Pitcher']['Height(inches)']

#plot_density_function(starting_pitchers, relief_pitchers)
#plot_histogram(starting_pitchers, relief_pitchers)




"ANALYSIS"

def check_normality(data):
    stat, p = shapiro(data)  # test for normality
    alpha = 0.05
    if p < alpha:  # null hypothesis: data comes from a normal distribution
        print('The null hypothesis can be rejected')
    else:
        print('The null hypothesis cannot be rejected')




def test_hypothesis(data1, data2):
    _, p = mannwhitneyu(data1, data2)
    alpha = 0.05
    print("p-value = {}".format(p))
    if p < alpha:
        print('The null hypothesis can be rejected')
    else:
        print('The null hypothesis cannot be rejected')



print("NULL HYPOTHESIS: Weight of starting_pitchers is normally distributed")
check_normality(starting_pitchers.values)

print("NULL HYPOTHESIS: Weight of relief_pitchers is normally distributed")
check_normality(relief_pitchers.values)

print('-----------------------------------')

print("NULL HYPOTHESIS: Weight cannot be used to distinguish players "
      "on Starting_Pitcher and Relief_Pitcher positions")
test_hypothesis(starting_pitchers, relief_pitchers)


