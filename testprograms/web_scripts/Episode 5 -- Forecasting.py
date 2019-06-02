
import math
import random
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
import seaborn as sns
from scipy import stats

# WRANGLING
mpl.rcParams['figure.figsize'] = (15,5)
get_ipython().run_line_magic('matplotlib', 'inline')
X = np.loadtxt("DataSets/RequestRates.csv", delimiter=",")[:,1]


# Request rates to nodes in a cluster
X = np.loadtxt("DataSets/ReqMultiNode.csv", delimiter=',', usecols=(0,1,2,3,4,5))[:1000]
T, A, B, C, D, E = X.T

# ANALYSIS

b, a, r_value, p_value, std_err = stats.linregress(A,B)

print "model = {} + {} * x".format(a,b)

grid = [0,1,2,3,4,5]
f    = lambda x: a + b*x

plt.scatter(A,B)
plt.plot(grid, map(f, grid))

