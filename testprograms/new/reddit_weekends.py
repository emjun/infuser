import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#import gzip
import difflib
import datetime
import time
from scipy import stats

"""
https://github.com/apaunescu/DS/blob/master/e5/reddit_weekends.py
RUNS FINE NOW, BUT HAD TO REWRITE A COUPLE OF LINES
"""

"WRANGLING"


def filterWeekend(date):
    if (date.weekday() == 5) or (date.weekday() == 6):
        return date
    return np.NaN

def filterWeekday(date):
    if (date.weekday() == 5) or (date.weekday() == 6):
        return np.NaN
    return date

def filterCanada(subreddit):
    if (subreddit != 'canada'):
        return np.NaN
    return subreddit


reddit_counts = 'reddit-counts.json'

counts = pd.read_json(reddit_counts, lines=True)
counts['date'] = pd.to_datetime(counts['date'])



counts = counts[(counts['date'].dt.year >= 2012) & (counts['date'].dt.year <= 2013)]
counts['subreddit'] = counts['subreddit'].apply(filterCanada)
weekends = counts[(counts['date'].dt.weekday == 5) | (counts['date'].dt.weekday == 6)]


# this line was added by Lukas
weekdays = counts[(counts['date'].dt.weekday != 5) & (counts['date'].dt.weekday != 6)]
"""
weekdays = pd.DataFrame(data = counts)
weekdays['date'] = counts['date']
weekdays['date'] = weekdays['date'].apply(filterWeekday)
"""


weekends = weekends.dropna()
weekdays = weekdays.dropna()

weekdays = weekdays.reset_index(drop = True)
weekends = weekends.reset_index(drop = True)


"ANALYSIS"


initialt = stats.ttest_ind(weekdays['comment_count'], weekends['comment_count'])[1]

initialweekday = (stats.normaltest(weekdays['comment_count']))[1]
initialweekend = (stats.normaltest(weekends['comment_count']))[1]


initialLevene = (stats.levene(weekdays['comment_count'], weekends['comment_count']))[1]

transformweekday = (stats.normaltest(np.sqrt(weekdays['comment_count'])))[1]
transformweekend = (stats.normaltest(np.sqrt(weekends['comment_count'])))[1]


transformLevene = (stats.levene(np.sqrt(weekdays['comment_count']), np.log(weekends['comment_count'])))[1]


# Infuser has trouble with the commented lines, so we rewrite it
weekendpair = weekends[['comment_count', 'date', 'subreddit']].copy()
weekdaypair = weekends[['comment_count', 'date', 'subreddit']].copy()
#weekendpair = pd.DataFrame(data = weekends)
#weekdaypair = pd.DataFrame(data = weekdays)


# also, infuser again doesn't like the following two lines
#https://stackoverflow.com/questions/29917931/python-define-starting-day-in-week-definition-in-isocalendar-or-strftime-or-els
#weekendpair['date'] = weekends['date'].apply(lambda x: str(x.isocalendar()[0]) + "/" + str(x.isocalendar()[1]))
#weekdaypair['date'] = weekdays['date'].apply(lambda x: str(x.isocalendar()[0]) + "/" + str(x.isocalendar()[1]))

weekendpair = weekendpair.groupby('date').mean().reset_index()
weekdaypair = weekdaypair.groupby('date').mean().reset_index()

weeklyWeekend = (stats.normaltest(weekendpair['comment_count']))[1]
weeklyWeekday = (stats.normaltest(weekdaypair['comment_count']))[1]
weeklyLevene = (stats.levene(weekdaypair['comment_count'], weekendpair['comment_count']))[1]

weeklyTtest = (stats.ttest_ind(weekendpair['comment_count'], weekdaypair['comment_count']))[1]

uTest = (stats.mannwhitneyu(weekdays['comment_count'], weekends['comment_count']))[1]
