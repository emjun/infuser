import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress

# WRANGLING
dataframe = pd.read_csv("scottish_hills.csv")

x = dataframe.Height
y = dataframe.Latitude

# ANALYSIS
stats = linregress(x, y)

m = stats.slope
b = stats.intercept

plt.scatter(x, y)
plt.plot(x, m * x + b, color="red")   # I've added a color argument here

plt.savefig("figure.png")