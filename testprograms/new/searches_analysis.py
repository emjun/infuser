import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#import gzip
import difflib
import datetime
import time
from scipy import stats
from scipy.stats import chi2_contingency
"""
https://github.com/apaunescu/DS/blob/fbecc6c9e73aaf4519bb52c3886f411bc9de2b35/e6/ab_analysis.py
WORKS FOR SOME PART OF THE FILE, ISSUES A WARNING
DOESNT WORK FOR SECOND PART OF ANALYSIS, INFUSER REJECTS CODE
"""

"WRANGLING"

searchdata_file = 'searches.json'

searches = pd.read_json(searchdata_file, lines=True)
odd_id = searches[(searches['uid'] %2 != 0)]
even_id = searches[(searches['uid'] %2 == 0)]
odds_searched = odd_id[(odd_id['search_count'] > 0)]
odd_unsearched = odd_id[(odd_id['search_count'] == 0)]

evens_searched = even_id[(even_id['search_count'] > 0)]
evens_unsearched = even_id[(even_id['search_count'] == 0)]


"ANALYSIS"


obs1 = np.array([[odds_searched.shape[0], odd_unsearched.shape[0]], [evens_searched.shape[0], evens_unsearched.shape[0]]])
chi = (chi2_contingency(obs1))
mannwhitneyu = stats.mannwhitneyu(odd_id['search_count'], even_id['search_count'])

"""
# INFUSER DOES NOT ACCEPT THE FOLLOWING LINES: UNABLE TO JUDGE TYPE FOR EXPRESSION
odds_searched = odds_searched[(odds_searched['is_instructor'] == True)]
odd_unsearched = odd_unsearched[(odd_unsearched['is_instructor'] == True)]



evens_searched = evens_searched[(evens_searched['is_instructor'] == True)]
evens_unsearched = evens_unsearched[(evens_unsearched['is_instructor'] == True)]

odd_id = odd_id[(odd_id['is_instructor'] == True)]
even_id = even_id[(even_id['is_instructor'] == True)]

obs1 = np.array([[odds_searched.shape[0], odd_unsearched.shape[0]], [evens_searched.shape[0], evens_unsearched.shape[0]]])
chi2 = (chi2_contingency(obs1))

mannwhitneyu2 = stats.mannwhitneyu(odd_id['search_count'], even_id['search_count'])
"""