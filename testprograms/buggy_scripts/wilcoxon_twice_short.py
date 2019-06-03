'''
buggy lines: 47
'''

import pandas as pd
from scipy import stats


def age_to_categories(df):
    df.loc[df['agegrp']=='30-45', 'agegrp'] = float(1);
    df.loc[df['agegrp']=='46-59', 'agegrp'] = float(2);
    df.loc[df['agegrp']=='60+', 'agegrp'] = float(3);
    return df


"WRANGLING"

df = pd.read_csv("../data/blood_pressure.csv")

df[['bp_before','bp_after']].describe()

df['bp_difference'] = df['bp_before'] - df['bp_after']

df['bp_difference'][df['bp_difference']==0]


df = age_to_categories(df)

men = df[df['sex'] == 'Male']
women = df[df['sex'] == 'Female']

men_age = men['agegrp']
women_age = women['agegrp']


"ANALYSIS"

# the first three should work
stats.wilcoxon(df['bp_difference'])

stats.wilcoxon(df['bp_before'], df['bp_after'])

stats.wilcoxon(men_age.append(women_age))

# while this one compares age categories with blood pressure that have nothing to 
# do with each other ==> should fail
stats.wilcoxon(men_age.append(women_age), df['bp_difference'])
