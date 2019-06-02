import pandas as pd
from scipy import stats

# WRANGLING

# must specify that blank space " " is NaN
experimentDF = pd.read_csv("../data/parasite_data.csv", na_values=[" "])


# show all entries in the ShannonDiversity column > 2.0
print (experimentDF[experimentDF["ShannonDiversity"] > 2.0])

print ("Mean virulence across all treatments:", stats.sem(experimentDF["Virulence"]))

print ("Mean Shannon Diversity w/ 0.8 Parasite Virulence =", experimentDF[experimentDF["Virulence"] == 0.8]["ShannonDiversity"].mean())


# select two treatment data sets from the parasite data
treatment1 = experimentDF[experimentDF["Virulence"] == 0.5]["ShannonDiversity"]
treatment2 = experimentDF[experimentDF["Virulence"] == 0.8]["ShannonDiversity"]


print ("Data set 1:\n", treatment1)
print ("Data set 2:\n", treatment2)


treatment3 = experimentDF[experimentDF["Virulence"] == 0.8]["ShannonDiversity"]
treatment4 = experimentDF[experimentDF["Virulence"] == 0.9]["ShannonDiversity"]

print ("Data set 3:\n", treatment3)
print ("Data set 4:\n", treatment4)


# ANALYSIS

# A RankSum test will provide a P value indicating whether or not the two
# distributions are the same.

z_stat, p_val = stats.ranksums(treatment1, treatment2)

print ("MWW RankSum P for treatments 1 and 2 =", p_val)


z_stat, p_val = stats.ranksums(treatment3, treatment4)

print ("MWW RankSum P for treatments 3 and 4 =", p_val)