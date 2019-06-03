import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
from scipy import stats

"WRANGLING"
# Request rates to nodes in a cluster
X = np.loadtxt("../data/forecasting.csv", delimiter=',', usecols=(0,1,2,3,4,5))[:1000]
T, A, B, C, D, E = X.T


"ANALYSIS"

b, a, r_value, p_value, std_err = stats.linregress(A,B)

grid = [1,2,3,4,5,6,7,8,9,10]
f    = lambda x: a + b*x
line = list(map(f, A))

plt.scatter(A,B)
plt.plot(grid, line)

