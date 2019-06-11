import pandas as pd
from scipy import stats

"WRANGLING" 
df = pd.read_csv("../data/blood_pressure.csv")

df[['bp_before','bp_after']].describe()

df['bp_difference'] = df['bp_before'] - df['bp_after']
df['bp_difference'][df['bp_difference']==0]


"ANALYSIS"
stats.wilcoxon(df['bp_difference'])

stats.wilcoxon(df['bp_before'], df['bp_after'])

