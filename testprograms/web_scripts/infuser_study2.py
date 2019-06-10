# Import librararies
import numpy as np
import csv as csv
import pandas as pd
import math
import json
import datetime as dt
import pprint
from IPython.display import display, HTML
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.sandbox.stats.multicomp import multipletests
import statsmodels.api as sm
from statistics import mean, stdev



"WRANGILNG"


# Set options
pd.set_option('display.max_columns', None)
pd.set_option("display.max_rows",200)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 200)

# Get datasets - make sure dates line up, or some participant ids will not exist in certian places
df_a_b = pd.read_csv('../data/study_data/footer_results_6_18_18.csv', engine='python')
df_motivations = pd.read_csv('../data/study_data/all_motivation_6_18_18.csv', engine='python')

df_dem_color = pd.read_csv('../data/study_data/demographics_color_6_18_18.csv', engine='python')
df_dem_implicit = pd.read_csv('../data/study_data/demographics_implicit_6_18_18.csv', engine='python')
df_dem_pm = pd.read_csv('../data/study_data/demographics_PM_6_18_18.csv', engine='python')
df_dem_memory = pd.read_csv('../data/study_data/demographics_memory_6_18_18.csv', engine='python')
df_dem_thinking_style = pd.read_csv('../data/study_data/demographics_thinking_style_6_18_18.csv', engine='python')
df_age_color = pd.read_csv('../data/study_data/color_real_age_6_18_18.csv', engine='python') 


df_slogans = pd.read_csv('../data/study_data/studythingy.csv', engine='python', sep=';')

### Merge color age results to get age of participants ###
df_dem_color = df_dem_color.merge(df_age_color[['participant_id', 'realAge']], on='participant_id')

### Format Data ###

# rename columns
df_a_b.columns = ['id', 'participant_id', 'data', 'timestamp']
df_motivations.columns = ['index', 'participant_id', 'study', 'data', 'timestamp', 'locale', 'timestamp_1', 'timestamp_2', 'satisficing' ]
df_slogans.columns = ['id', 'slogan', 'contributor_id', 'design_id']
df_dem_pm = df_dem_pm.rename(index=str, columns={"current_time":"timestamp", "country0":"country"})
df_dem_implicit = df_dem_implicit.rename(index=str, columns={"participantId": "participant_id", "current_time":"timestamp", "country0":"country"})
df_dem_color = df_dem_color.rename(index=str, columns={"time_stamp":"timestamp", "current_ctry":"country", "realAge":"age"})
df_dem_memory = df_dem_memory.rename(index=str, columns={"current_time":"timestamp", "country0":"country"})
df_dem_thinking_style = df_dem_thinking_style.rename(index=str, columns={"country0":"country"})


# Add column to slogans for study name
df_slogans = df_slogans.drop(['contributor_id'], axis=1)
df_slogans['study'] = df_slogans.design_id.apply(lambda x: "Implicit Memory" if x > 12 else ("Thinking Style" if x > 6 else "Color"))


# convert json columns to dicts 
df_a_b.data = df_a_b.data.apply(lambda x: json.loads(x))
df_motivations.data = df_motivations.data.apply(lambda x: json.loads(x))

# clean slogan col to just hold slogan string
df_slogans.slogan = df_slogans.slogan.apply(lambda x: json.loads(str.strip(x, '\'')))
df_slogans.slogan = df_slogans.slogan.apply(lambda x: x['slogan'])

# convert timestamps to datetimes
# clear of any timestamps that are 0 first in df_motivations (I think I can ignore this data, but check)
df_motivations = df_motivations[df_motivations['timestamp'] != '0000-00-00 00:00:00']
df_a_b['timestamp'] = pd.to_datetime(df_a_b['timestamp'])
df_motivations['timestamp'] = pd.to_datetime(df_motivations['timestamp'])
df_dem_color['timestamp'] = pd.to_datetime(df_dem_color['timestamp'])
df_dem_implicit['timestamp'] = pd.to_datetime(df_dem_implicit['timestamp'])
df_dem_pm['timestamp'] = pd.to_datetime(df_dem_pm['timestamp'])
df_dem_memory['timestamp'] = pd.to_datetime(df_dem_memory['timestamp'])


# Add study name column for each demographic df
df_dem_color['study_name'] = 'color_age'
df_dem_implicit['study_name'] = 'implicit_memory'
df_dem_memory['study_name'] = 'memory'
df_dem_pm['study_name'] = 'perceptual_models'
df_dem_thinking_style['study_name'] = 'analytic'

# Set indices as participant id #
df_a_b = df_a_b.set_index('participant_id', drop=False)
df_motivations = df_motivations.set_index('participant_id', drop=False)


### Clean + Parse Data ###

# Drop uneccesary columns
df_motivations = df_motivations.drop(['timestamp_1', 'timestamp_2', 'satisficing', 'index'], axis=1)

# Drop rows in motivations and demographics 
# with a timestamp before motivation survey was redeployed (before Febr, 2018)

cut_off_date = dt.datetime(2018, 2, 1) # Original


df_motivations = df_motivations[df_motivations['timestamp'] > cut_off_date]
df_dem_color = df_dem_color[df_dem_color['timestamp'] > cut_off_date]
df_dem_implicit = df_dem_implicit[df_dem_implicit['timestamp'] > cut_off_date]
df_dem_pm = df_dem_pm[df_dem_pm['timestamp'] > cut_off_date]
df_dem_memory = df_dem_memory[df_dem_memory['timestamp'] > cut_off_date]



# Drop any rows in a/b testing without a participant id and all first round of slogan testing (before setups)
df_a_b = df_a_b[df_a_b['participant_id'] != 0]
df_a_b = df_a_b[df_a_b['data'].apply(lambda x: ('slogan' not in x))]

# Get only participants from motivations who finished (i.e. saw a the footer)
df_mot_finished = df_motivations[df_motivations['participant_id'].isin(df_a_b['participant_id'])].copy()

# Seperate the motivation data column into full df, seperate again only the scores (below)
df_motivation_data = df_mot_finished['data'].apply(pd.Series).copy()

# Convert motivations to numeric
df_motivation_data[['bored', 'compare', 'selfLearn', 'science', 'fun']] = df_motivation_data[['bored', 'compare', 'selfLearn', 'science', 'fun']].apply(pd.to_numeric)


print('Number of motivations before taking out people who did not answer at least one:' , len(df_motivation_data))

# Drop any people who put nothing for any motivation (very conservative for now)
df_motivation_data = df_motivation_data.dropna(subset=['bored', 'compare', 'selfLearn', 'science', 'fun'], how='any')
# df_motivation_data_scores = df_motivation_data[['bored', 'compare', 'selfLearn', 'science', 'fun']]

print('Number of motivations after taking out people who did not answer at least one:' , len(df_motivation_data))

# Drop any rows with participant id not with full motivations
df_a_b = df_a_b[df_a_b['participant_id'].isin(df_motivation_data['participant_id'])]

# seperate clickthroughs and setup data
df_clickthrough = df_a_b[df_a_b['data'].apply(lambda x: ((x['data_type'] == 'tracking:a_b_clickthrough')))].copy()
df_setups = df_a_b[df_a_b['data'].apply(lambda x: x['data_type'] == 'tracking:setup')].copy()

# Drop any duplicates in df_clickthrough
df_clickthrough = df_clickthrough.drop_duplicates('participant_id')


# drop any participants without a real age given on the color age test
df_dem_color[df_dem_color['age'] == 0] = np.nan
# df_dem_color = df_dem_color[df_dem_color['age'] > 0]


# In each study, get only the participants who have a corrosponding id in the clickthroughs 
df_dem_color = df_dem_color[df_dem_color['participant_id'].isin(df_motivation_data[df_motivation_data['study_name'] == 'color_age']['participant_id'])]
df_dem_memory = df_dem_memory[df_dem_memory['participant_id'].isin(df_motivation_data[df_motivation_data['study_name'] == 'memory']['participant_id'])]
df_dem_pm = df_dem_pm[df_dem_pm['participant_id'].isin(df_motivation_data[df_motivation_data['study_name'] == 'perceptual_models']['participant_id'])]
df_dem_implicit = df_dem_implicit[df_dem_implicit['participant_id'].isin(df_motivation_data[df_motivation_data['study_name'] == 'implicit_memory']['participant_id'])]
df_dem_thinking_style = df_dem_thinking_style[df_dem_thinking_style['participant_id'].isin(df_motivation_data[df_motivation_data['study_name'] == 'analytic_test']['participant_id'])]


# Convert Perceptual Models df to fit with rest of gender setup 
df_dem_pm.loc[df_dem_pm['gender'] == 1,'gender'] = 4
df_dem_pm.loc[df_dem_pm['gender'] == 2,'gender'] = 5
df_dem_pm.loc[(df_dem_pm['gender'] == 0) | (df_dem_pm['gender'] == 3),'gender'] = 2


df_dem_pm.loc[df_dem_pm['gender'] == 4,'gender'] = 0
df_dem_pm.loc[df_dem_pm['gender'] == 5,'gender'] = 1



# Make a full demographics df
df_dem_full = pd.concat([df_dem_color[['gender', 'age', 'country', 'participant_id', 'study_name']],
            df_dem_implicit[['gender', 'age', 'country', 'participant_id', 'study_name']],
            df_dem_memory[['gender', 'age', 'country', 'participant_id', 'study_name']],
            df_dem_pm[['gender', 'age', 'country', 'participant_id', 'study_name']],
            df_dem_thinking_style[['gender', 'age', 'country', 'participant_id', 'study_name']]])



# Set participant id to numeric and set as index
df_dem_full['participant_id'] = pd.to_numeric(df_dem_full['participant_id'])
df_dem_full = df_dem_full.set_index('participant_id', drop=False)


df_motivation_data['participant_id'] = pd.to_numeric(df_motivation_data['participant_id']) 

### Sync datasets ###
# Just some housekeeping, get all the datasets with correct amounts in them

# Get motivations and demographics in same place
df_mot_dem_data = df_dem_full.merge(df_motivation_data, on='participant_id', how='right')

print('length of other gender :', len(df_mot_dem_data[df_mot_dem_data['gender'] == 2]))
# drop people who put other for gender
df_mot_dem_data = df_mot_dem_data[df_mot_dem_data['gender'] <= 1]


print("length of demographics: ", len(df_dem_full))
print(df_dem_full.dtypes)
print("length of motivations: ", len(df_motivation_data))
print(df_motivation_data.dtypes)
print("length of them together: ", len(df_mot_dem_data))


# convert all participant ids to numeric
df_clickthrough['participant_id'] = pd.to_numeric(df_clickthrough['participant_id'])
df_mot_dem_data['participant_id'] = pd.to_numeric(df_mot_dem_data['participant_id'])

# Drop duplicates 
df_clickthrough= df_clickthrough.drop_duplicates('participant_id')
df_setups = df_setups.drop_duplicates('participant_id')
df_mot_dem_data= df_mot_dem_data.drop_duplicates('participant_id')

# Get only setup rows that relate to a participant who clicked on a slogan
df_clickthrough_setups = df_setups[df_setups['participant_id'].isin(df_clickthrough['participant_id'])].copy()

# Get only set up rows that relate to a participant who did NOT click on a slogan
df_no_clickthrough_setups = df_setups.drop(df_clickthrough.index).copy()


df_clickthrough_data = df_clickthrough.data.apply(pd.Series)
df_setup_data = df_setups.data.apply(pd.Series)
df_no_clickthrough_data = df_no_clickthrough_setups.data.apply(pd.Series)

# rename study_name_x to study_name, drop study_name_x
df_mot_dem_data = df_mot_dem_data.drop('study_name_x', axis=1)
df_mot_dem_data = df_mot_dem_data.rename(index=str, columns={'study_name_y':'study'})



def set_up_framings(slogans, index):
    science_slogans = [4, 6, 10, 12, 14, 18]
    learn_slogns = [1, 9, 16]
    fun_bored_slogans = [2, 3, 8, 13, 15]
    # bored_slogans = []
    compare_slogans = [5, 7, 11, 17]
    
    for s in slogans.iterrows():
        if s[1]['design_id'] in science_slogans:
            slogans.at[s[1][index], 'framing'] = 'science_framing'
        elif s[1]['design_id'] in learn_slogns:
            slogans.at[s[1][index], 'framing'] = 'self_learn_framing'
        elif s[1]['design_id'] in fun_bored_slogans:
            slogans.at[s[1][index], 'framing'] = 'fun_bored_framing'
        elif s[1]['design_id'] in compare_slogans:
            slogans.at[s[1][index], 'framing'] = 'compare_framing'
        

df_slogans = df_slogans.set_index('slogan', drop=False)
set_up_framings(df_slogans, 'slogan')






"ANALYSIS"

## Helper functions ### 

def check_assumptions(groups, entire_df):
    # Levene's test for homogenity of variance
    w, p = stats.levene(*groups)
    print("------------------------------------------------------")
    print('Levene\'s test for homogeniety of variance (p > .05 -> variance is equal)')
    print('w =', ('%.2f' % w), 'p =', ('%.4f' % p))
    print("------------------------------------------------------")
    
    # Shapiro-Wilk test for normal distribution
    f, p = stats.shapiro(entire_df)
    print("------------------------------------------------------")
    print('Shapiro-Wilk test for normal distribution (p > .05 -> drawn from normal distribution)')
    print('w =', ('%.2f' % w), 'p =', ('%.4f' % p))
    print("------------------------------------------------------")
    
    
# Calculating effect size 
# taken from https://stackoverflow.com/questions/15436702/estimate-cohens-d-for-effect-size?rq=1
def cohens_d(x, y):
    lx = len(x) - 1
    ly = len(y)- 1
    md = abs(x.mean() - y.mean())        ## mean difference (numerator)
    pld_var = lx * x.var() + ly * y.var()
    pld_var = pld_var/(lx + ly)
    pld_var = np.sqrt(pld_var)
    print('Cohen\'s d:', md/pld_var )
    return cohens_d


def descriptive_stats(groups):
    print("------------------------------------------------------")
    print('Descriptive Stats')
    for k in groups.keys():
        print(k, ':')
        mot_list = np.asarray(groups[k])
        mot_list = mot_list[~np.isnan(mot_list)]
        print('mean =', ('%.2f' % mot_list.mean()) , 'sd =', ('%.2f' % mot_list.std()))
        print('n =', len(mot_list))
    print("------------------------------------------------------")
    
# function for running tukey post hoc tests for all keys in dict for a single motivation
# assumes same format for dict as function run_anova
def run_tukey(df, mot, groups):
    df_no_na = df.dropna(subset=[mot])
    results = pairwise_tukeyhsd(endog=df_no_na[mot],     # Data
                              groups=df_no_na[groups],   # Groups
                              alpha=0.05)          # Significance level
    # convert tukey to df 
    # Taken from https://stackoverflow.com/questions/40516810/saving-statmodels-tukey-hsd-into-a-python-panda-dataframe
    df_tukey = pd.DataFrame(data=results._results_table.data[1:], columns=results._results_table.data[0])
    # get only tests where reject is true 
    df_tukey_true = df_tukey[df_tukey['reject'] == True]
    df_tukey = df_tukey.sort_values(by='meandiff')
    print("------------------------------------------------------")
    print('Tukey results where reject is true:')
    display(df_tukey)
    print("------------------------------------------------------")
    
    for index, row in df_tukey.iterrows():
        c_one = df_no_na[df_no_na[groups] == row.group1][mot]
        c_two = df_no_na[df_no_na[groups] == row.group2][mot]
        print(row.group1, row.group2)
        cohens_d(c_one, c_two)  
    
def run_anova_groups(df, mot_ids, grp_id):
    grpd_df = df.groupby(grp_id)
    groups = list(grpd_df.groups.keys())
    measures = []
    for mot in mot_ids:
        print(mot)
        measures = {g:grpd_df.get_group(g)[mot].values for g in groups}
        measures = {g:measures[g][~np.isnan(measures[g])] for g in groups}
        f, p = stats.f_oneway(*measures.values())
        print('F =', ('%.3f' % f), 'p =', ('%.3f' % p))
        if p < .1:
            descriptive_stats(measures)
    
    
def run_just_stats(df, mot_ids, grp_id):
    grpd_df = df.groupby(grp_id)
    groups = list(grpd_df.groups.keys())
    measures = []
    for mot in mot_ids:
        print(mot)
        measures = {g:grpd_df.get_group(g)[mot].values for g in groups}
        measures = {g:measures[g][~np.isnan(measures[g])] for g in groups}
        descriptive_stats(measures)


def run_t_test_groups(df, mot_ids, grp_id):
    t_test_p_vals = [] 
    grpd_df = df.groupby(grp_id)
    groups = list(grpd_df.groups.keys())
    measures = []
    for mot in mot_ids:
        print(mot)
        measures = {g:grpd_df.get_group(g)[mot].values for g in groups}
        measures = {g:measures[g][~np.isnan(measures[g])] for g in groups}
        degrees_of_freedom = (len(measures[groups[0]]) - 1) + (len(measures[groups[1]]) - 1)
        t, p = stats.ttest_ind(*measures.values())
        t_test_p_vals.append(p)
        print('t{', degrees_of_freedom ,'}=', ('%.3f' % t), 'p =', ('%.3f' % p))
        if (p < .1):
            run_tukey(df, mot, grp_id)
            descriptive_stats(measures)
        cohens_d(measures[groups[0]], measures[groups[1]])
    r, p_vals, sidak, bonferroni = multipletests(t_test_p_vals, method='fdr_bh')

    print('Corrected p-values and rejections')
    print('--------------------------------------------------------')
    print('Reject the null? ', r)
    
    print(mot_ids)
    print('Corrected p values: ', p_vals)






### Descriptive Stats ###

def describe_participants(df, study):
    df_study = df[df['study'] == study].copy()
    gender = df_study['gender'].dropna()

    print('------------------------------------------------------------------------------' )
    print(study, ' total participants: ', len(df_study))
    print('------------------------------------------------------------------------------' )
    print('------------------------------------------------' )
    print('% of female: ', (100 * (len(gender[gender == 1])/len(gender))))
    print('% of male: ', (100 * (len(gender[gender == 0])/len(gender))))
    print('Average Age: ', df_study['age'].dropna().mean(), '(', ('%.2f' % df_study['age'].dropna().std()) ,')')
    
    
# Get average for motivations
# For all studies combined:
# total_dem_length = len(df_dem_color) + len(df_dem_implicit) + len(df_dem_memory) + len(df_dem_pm) + len(df_dem_thinking_style)
print('------------------------------------------------------------------------------' )
print('ALL STUDIES total participants: ', len(df_mot_dem_data))
print('------------------------------------------------------------------------------' )
print('Descriptive stats - Motivations')
print('------------------------------------------------' )
print('Mean motivations:')
print(df_mot_dem_data[['bored', 'selfLearn', 'fun', 'science', 'compare']].dropna().astype(int).mean())
print('------------------------------------------------' )
print('Std motivations:')
print(df_mot_dem_data[['bored', 'selfLearn', 'fun', 'science', 'compare']].dropna().astype(int).std())

for s in df_mot_dem_data['study'].unique():
    describe_participants(df_mot_dem_data, s)






### get count of how many times each slogan was shown, store in df
df_slogans['shown_first'] = 0
df_slogans['shown_second'] = 0
df_slogans['select_first'] = 0
df_slogans['select_second'] = 0
df_slogans['select_total'] = 0

df_clickthrough_setups['participant_id'] = pd.to_numeric(df_clickthrough_setups['participant_id'])
# get count for number of times each slogan appears
for setup in df_setups.iterrows():
    # get prev 1st count for 1st slogan, and 2nd count for 2nd slogan
    prev_count_first = df_slogans.get_value(setup[1]['data']['slogan_1'], 'shown_first')
    prev_count_second = df_slogans.get_value(setup[1]['data']['slogan_2'], 'shown_second')

    # update counts
    df_slogans.set_value(setup[1]['data']['slogan_1'], 'shown_first', prev_count_first + 1) 
    df_slogans.set_value(setup[1]['data']['slogan_2'], 'shown_second', prev_count_second + 1)
    
# get count for number of times each slogan was chosen 
for click in df_clickthrough.iterrows():
    setup = df_clickthrough_setups.loc[click[1]['participant_id'],'data']
    # Check if slogan that was clicked on appeared on top or bottom
    if click[1]['data']['chosen_slogan'] == setup['slogan_1']:
        prev_select_first = df_slogans.get_value(click[1]['data']['chosen_slogan'], 'select_first')
        df_slogans.set_value(click[1]['data']['chosen_slogan'], 'select_first', prev_select_first + 1) 
       
    if click[1]['data']['chosen_slogan'] == setup['slogan_2']: 
        prev_select_second = df_slogans.get_value(click[1]['data']['chosen_slogan'], 'select_second')
        df_slogans.set_value(click[1]['data']['chosen_slogan'], 'select_second', prev_select_second + 1) 
        
    # add to select total seperately in order to check that numbers match 
    prev_select = df_slogans.get_value(click[1]['data']['chosen_slogan'], 'select_total')
    df_slogans.set_value(click[1]['data']['chosen_slogan'], 'select_total', prev_select + 1) 

# Get total number of times a slogan was shown
df_slogans['shown_total'] = df_slogans['shown_first'] + df_slogans['shown_second']

df_slogans['percent_clicked'] = df_slogans['select_total'] / df_slogans['shown_total']



# Reset index of slogans to the slogan itself (making it dict-like) ### 
df_slogans = df_slogans.set_index('slogan', drop=False)

display(df_slogans.groupby('framing')['select_total'].sum())

for study in df_slogans.study.unique():
    display(df_slogans[df_slogans['study'] == study].sort_values(by='percent_clicked'))
    
    
    
    
    
### Clickthrough rate and additional formatting for clickthrough dfs ###

# clean clickthroughs to only contain rows that have a corrosponding set up
df_clickthrough = df_clickthrough[df_clickthrough['participant_id'].isin(df_clickthrough_setups['participant_id'])]

# clean clickthroughs of duplicates 
df_clickthrough = df_clickthrough.drop_duplicates('participant_id')

print('rate of clickthrough (selecting either of the two slogans):', 100 * len(df_clickthrough)/len(df_setups))

# Check what percentage of the chosen slogans are the top one (i.e do people just select the top slogan)
count_first_slogans = 0
for click in df_clickthrough.iterrows():
    setup = df_clickthrough_setups.loc[click[1]['participant_id'],'data']    
    if click[1]['data']['chosen_slogan'] == setup['slogan_1']:
        count_first_slogans += 1
        
print(len(df_clickthrough))
print('Percent of first slogans clicked:', ('%.2f' % (100 * df_slogans['select_first'].sum()/(df_slogans['select_total'].sum()))), '%')

print('meam clickthrough percent =', df_slogans['percent_clicked'].mean(), 'sd =', df_slogans['percent_clicked'].std())





### Set up DF with all clickthroughs and participant demographics/motivations ###

# Set up framings
for f in df_slogans.framing.unique():
    df_setup_data.loc[df_setup_data['slogan_1'].apply(lambda x: df_slogans.loc[x]['framing'] == f), f] = 1
    df_setup_data.loc[df_setup_data['slogan_2'].apply(lambda x: df_slogans.loc[x]['framing'] == f), f] = 1

df_clickthrough_data['click'] = df_clickthrough_data['chosen_slogan'].apply(lambda x: df_slogans.loc[x]['framing'])

df_clickthrough_data = df_clickthrough_data.rename(index=str, columns={'chosen_slogan':'click_slogan'})
                                                   
df_setup_click_data = df_setup_data.merge(df_clickthrough_data[['participant_id', 'click', 'click_slogan']], on='participant_id', how='left')

df_mot_dem_data['participant_id'] = pd.to_numeric(df_mot_dem_data['participant_id'])
df_setup_click_data['participant_id'] = pd.to_numeric(df_setup_click_data['participant_id'])

df_mot_dem_setup_click_data = df_mot_dem_data.drop('timestamp', axis=1).merge(df_setup_click_data.drop(['data_type','timestamp'], axis=1), on='participant_id', how=
                                                                             'left')

df_mot_dem_setup_click_data = df_mot_dem_setup_click_data.dropna(subset=['gender', 'age', 'bored', 'fun', 'selfLearn', 'science', 'compare'], how='any')









### Make figure 5

import matplotlib.pyplot as plt

grp_frame = df_slogans.groupby('framing')
labels = df_slogans['design_id']
frames = ['Self-learn', 'Fun & Bored', 'Science', 'Compare']
nb_colors = len(plt.rcParams['axes.prop_cycle'])
colors = ['b', 'g', 'r', 'c']
symbols = [(3, 0, 0), '*', '+', (0, 3, 0)]

plt.figure(num=None, figsize=(15, 1), dpi=200)

color_map = {f:colors.pop() for f in df_slogans.framing.unique()}
symbols_map = {f:symbols.pop() for f in df_slogans.framing.unique()}

plots = []

for f in df_slogans.framing.unique(): 
    frame_group = grp_frame.get_group(f)
    y = [1 for f in frame_group['design_id']]
    x = frame_group['percent_clicked'] * 100
    p = plt.scatter(x, y, c=color_map[f], marker=symbols_map[f])
    plots.append(p)
        
plt.legend(plots,
           frames,
           scatterpoints=1,
           loc='lower left',
           bbox_to_anchor=(0,-1),
           ncol=3,
           fontsize=12)
    

plt.axes().get_yaxis().set_ticks([])
plt.margins(0.05)
plt.xlabel('percent clicked', fontsize=15)

# Tweak spacing to prevent clipping of tick-labels
plt.subplots_adjust(bottom=0.15)
plt.show()





# look at age and motivations across people who clicked through and didn't

dem_and_mot = df_mot_dem_data.columns.drop(['participant_id', 'study', 'timestamp', 'country']).tolist()

print('Click vs No')
print('------------------------------------------------------------')
df_clicked_vs_not = df_mot_dem_setup_click_data.copy()
df_clicked_vs_not['click'] = df_clicked_vs_not['click'].fillna('None')
df_clicked_vs_not.loc[df_clicked_vs_not['click'] != 'None', 'click'] = 'just_click'
dem_and_mot.remove('gender')
run_t_test_groups(df_clicked_vs_not, dem_and_mot, 'click')
print('------------------------------------------------------------')





### ANOVA over framings -- did the motivations of participants significantly differ across who clicked on what frame? ### 

dem_and_mot = df_mot_dem_data.columns.drop(['participant_id', 'study', 'timestamp', 'country']).tolist()

print('Descriptive stats of each frames')
print('------------------------------------------------------------')
df_clicked = df_mot_dem_setup_click_data.dropna(subset=['click'], how='any').copy()
print(len(df_clicked))
dem_and_mot.remove('gender')
run_anova_groups(df_clicked, dem_and_mot, 'click')
print('------------------------------------------------------------')





## area for checking specific motivations across frames -- looking at the bored motivation since the only one with a close to significant value 
# Now checking if bored does differ significantly across any two groups 
# Run multiple testing correction for all tests 

def run_t_test(values_1, values_2, value_1_name, value_2_name):
    f, p = stats.ttest_ind(values_1,values_2)
    degrees_of_freedom = (len(values_1) - 1) + (len(values_2) - 1)
    effect_size = cohens_d(values_1, values_2)
    print(value_1_name, ':' , values_1.mean(), values_1.std())
    print(value_2_name, ':', values_2.mean(), values_2.std())
    print('$t_{' ,degrees_of_freedom , '} =', ('%.2f' % f), '$', 'p =', '$', ('%.3f' % p), '$','d=', '$', effect_size, '$')
    return p

p_values = []
grpd_df = df_clicked.groupby('click')
values_1 = df_clicked[df_clicked['click'] == 'self_learn_framing'].bored.values
values_2 = df_clicked[df_clicked['click'] == 'compare_framing'].bored.values
p_values.append(run_t_test(values_1, values_2, 'self_learn', 'compare'))


p_vals = []
grpd_df = df_clicked.groupby('click')
values_2 = df_clicked[df_clicked['click'] == 'fun_bored_framing'].bored.values
p_values.append(run_t_test(values_1, values_2, 'self_learn', 'fun_bored'))


p_vals = []
grpd_df = df_clicked.groupby('click')
values_2 = df_clicked[df_clicked['click'] == 'science_framing'].bored.values
p_values.append(run_t_test(values_1, values_2, 'self_learn', 'science'))


multipletests(p_values, method='fdr_bh')

   