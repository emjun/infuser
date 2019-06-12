# Code for converting a csv with JSON column into nice df in pandas
import csv as csv
import pandas as pd
import math
import json
import numpy as np
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import datetime as dt

"""
SAM: CHECK OUT LINE 42
"""

from functools import reduce
from statsmodels.formula.api import ols
from IPython.display import display


"WRANGLING"

#######################################
####### Functs for getting data  ######
#######################################

# convert json columns to dicts 
def decode_json_entry(json_db_entry):
    json_db_entry = json_db_entry.replace('\\"', '').replace('\\', '')
    try:
        decoded_json_entry = json.loads(json_db_entry)
    except json.JSONDecodeError:
        print("Error!",json_db_entry )
    return decoded_json_entry

def get_key(entry, val):
    return 'data_type' in entry and val == entry["data_type"];

def return_key_df(df, key):

    def l1(x):
        return get_key(x, key)
    return df[df.data.apply(l1)]['data'].apply(pd.Series).drop('data_type', axis=1)

"""
# the problem is the lambda function I guess
def return_key_df(df, key):
    return df[df.data.apply(lambda x: get_key(x, key))]['data'].apply(pd.Series).drop('data_type', axis=1)
"""

############################
####### Get the data  ######
############################

df = pd.read_csv('../data/litw_iq_data.csv', sep=";", engine='python')

# this line again is problematic
df.data = df.data.apply(decode_json_entry)

#df_data = df[df.data.apply(lambda x: get_key(x, 'study:data'))]['data'].apply(pd.Series).drop('data_type', axis=1)

'''
# Collect everything into one df, get some other useful vars
attention_checks_df = df[df.data.apply(lambda x: get_key(x, 'attentionCheck'))]['data'].apply(pd.Series).drop('data_type', axis=1)
formality_survey_df = df[df.data.apply(lambda x: get_key(x, 'formalitySurvey'))]['data'].apply(pd.Series).drop('data_type', axis=1)
formality_level_df = df[df.data.apply(lambda x: get_key(x, 'formalityLevel'))]['data'].apply(pd.Series).drop('data_type', axis=1)
# use only formal/informal version
formality_level_df = formality_level_df[formality_level_df['formality_level'] != 'normal']

values_df = df[df.data.apply(lambda x: get_key(x, 'valueSurvey'))]['data'].apply(pd.Series).drop('data_type', axis=1)
values_df = values_df.rename(index=str, columns={'self-direction': 'self_direction'})

df_correct = df_data.groupby('uuid')['is_correct'].describe()
df_correct = df_correct[df_correct['count'] == 32]
df_correct = df_correct.reset_index().drop(['unique', 'top', 'count'], axis=1)
# rename correct column and drop count col
df_correct = df_correct.rename(index=str, columns={"freq": "score"})

print('length of people who did all 32 rounds (2 practice): ', len(df_correct))


# get drop out column
df_checkpoints = df[df.data.apply(lambda x: get_key(x, 'tracking:checkpoint'))]['data'].apply(pd.Series).drop('data_type', axis=1)
df_checkpoints = df_checkpoints.groupby('uuid')['dropoutCode'].describe().reset_index()[['uuid', 'count']]
df_checkpoints['dropout'] = df_checkpoints['count'] < 41 

# # Getting clickthrough data -- Taken out since not using this right now, will need to download new data
# df_a_b = pd.read_csv('study_data/click_through.csv', engine='python')
# df_a_b.columns = ['id', 'participant_id', 'data', 'timestamp']
# df_a_b.data = df_a_b.data.apply(lambda x: json.loads(x))
# df_a_b['timestamp'] = pd.to_datetime(df_a_b['timestamp'])
# df_a_b = df_a_b.data.apply(pd.Series)
# df_a_b = df_a_b[df_a_b['study_name'] == 'spatial_awareness']

# merge to formality rating
df_formality_dropout = formality_level_df.merge(df_checkpoints, on='uuid')

# get demographics 
df_demographics = df[df.data.apply(lambda x: get_key(x, 'study:demographics'))]['data'].apply(pd.Series).drop('data_type', axis=1)
df_demographics[['education', 'age']] = df_demographics[['education', 'age']].apply(pd.to_numeric)

values = list(values_df.drop('uuid', axis=1).columns)

# convert values to numeric form
values_df[values] = values_df[values].apply(pd.to_numeric)

# formulas for coservation, and self-transcendence, etc.
values_df['conservation'] = .92 + (.15 * values_df['power']) + (.03 * values_df['achievement']) - (.17 * values_df['hedonism']) - (.25 * values_df['stimulation']) - (.31 * values_df['self_direction']) - (.26 * values_df['universalism']) + (.04 * values_df['benevolence']) + (.30 * values_df['tradition']) + (.30 * values_df['conformity']) + (.20 * values_df['security'])
values_df['self_transcendence'] = -.56 - (.30 * values_df['power']) - (.33 * values_df['achievement']) - (.16 * values_df['hedonism']) - (.14 * values_df['stimulation']) + (.04 * values_df['self_direction']) + (.22 * values_df['universalism']) + (.24 * values_df['benevolence']) + (.12 * values_df['tradition']) + (.03 * values_df['conformity']) + (.03 * values_df['security'])

# make comments df
df_comments = return_key_df(df, 'study:comments')

# merge all the dfs
dfs = [df_correct, attention_checks_df, formality_survey_df, formality_level_df, values_df, df_comments]
df_formality_values = reduce(lambda left,right: pd.merge(left,right,on='uuid'), dfs)

# convert values to numeric form
df_formality_values[['formality_rating', 'appropriate_rating', 'score']] = df_formality_values[['formality_rating', 'appropriate_rating', 'score']].apply(pd.to_numeric)

# even though TECHNICALLY true and false are 1/0, the logit function doesn't seem to know that
# so this is a way of fixing that.
df_formality_values.replace(to_replace={'engaged' : {True: '1', False: '0'}}, inplace = True)
df_formality_values['engaged'] = df_formality_values['engaged'].apply(pd.to_numeric)

# same for dropout
df_formality_dropout.replace(to_replace={'dropout' : {True: '1', False: '0'}}, inplace = True)
df_formality_dropout['dropout'] = df_formality_dropout['dropout'].apply(pd.to_numeric)

##############################
####### make a final df ######
##############################
df_dem_formality_values = df_demographics[['age', 'education', 'gender', 'uuid', 'retake', 'country0']].merge(df_formality_values, on='uuid')
# add in english variable
df_dem_formality_values['english'] =  df_dem_formality_values['country0'].apply(lambda x: x in ['United States', 'United Kingdom', 'Canada', 'Australia', 'Ireland', 'New Zealand', 'American Samoa'])

# convert gender to male/female
df_dem_formality_values['gender'] = df_dem_formality_values['gender'].replace({'1':'female', '0':'male'})


##############################
##### Add in turker stuff ####
##############################

# Turkers who started
df_turker = return_key_df(df, 'MTurk')
print("Turkers who started:", len(df_turker))

# Turkers who finished
df_turker_id = return_key_df(df, 'MturkWorkerID')

# Only take those those also submitted through mturk - no
# df_turk_results_id_1 = pd.DataFrame.from_csv('study_data/MTurk_results_1.csv')
# df_turk_results_id_2 = pd.DataFrame.from_csv('study_data/MTurk_results_2.csv')

# df_turk_results_id = pd.concat([df_turk_results_id_1,df_turk_results_id_2])
# df_turk_results_id = df_turk_results_id.rename(index=str, columns={'Answer.surveycode':'MTurkworkerID'})

# df_turker_id = df_turker_id.merge(df_turk_results_id[['MTurkworkerID']], on='MTurkworkerID')

# df_turker_id = df_turker_id.merge(df_turk_results_id, on='MTurkworkerID')
print("Turkers who finished:", len(df_turker_id))

# make new df of turker and formality/values (backwards compatability on the turker notebook)
df_dem_formality_turker = df_turker.merge(df_dem_formality_values, on='uuid', how='right')
df_dem_formality_turker['Turker'] = df_dem_formality_turker['Turker'].fillna(False)

# updated main df with turker info
df_dem_formality_values = df_turker.merge(df_dem_formality_values, on='uuid', how='right')
df_dem_formality_values['Turker'] = df_dem_formality_values['Turker'].fillna(False)

# Fill NaNs with false for both (above and below)

# Same for dropout

df_formality_turker_dropout = df_turker.merge(df_formality_dropout, on='uuid', how='right')
df_formality_turker_dropout['Turker'] = df_formality_turker_dropout['Turker'].fillna(False)


df_formality_dropout = df_turker.merge(df_formality_dropout, on='uuid', how='right')
df_formality_dropout['Turker'] = df_formality_dropout['Turker'].fillna(False)

##############################
##### End Turker Addition ####
##############################

##############################
######## Getting Time ########
##############################

# Get end of study (comments)
comments = df[df.data.apply(lambda x: get_key(x, 'study:comments'))]
checkpoints = df[df.data.apply(lambda x: get_key(x, 'tracking:checkpoint'))]

# Get beginning of study (IRB)
irb_checkpoint = checkpoints[checkpoints.data.apply(lambda x: x['description'] == 'irb')]

# Get uuid as own column
comments['uuid'] = comments.data.apply(pd.Series)['uuid']
irb_checkpoint['uuid'] = irb_checkpoint.data.apply(pd.Series)['uuid']

# convert the time stamp on both
comments['timestamp'] = pd.to_datetime(comments['timestamp'])
irb_checkpoint['timestamp'] = pd.to_datetime(irb_checkpoint['timestamp'])

# make a time dataframe with length of study
time_df = pd.merge(comments, irb_checkpoint, on='uuid')
time_df['time_diff'] = time_df['timestamp_x'] - time_df['timestamp_y']

# add time to take test to main df
df_dem_formality_values = time_df[['uuid', 'time_diff']].merge(df_dem_formality_values, on='uuid')
df_dem_formality_values['time_diff'] = df_dem_formality_values['time_diff'].apply(lambda x: (x.total_seconds())/60)

##############################
######### End Time ########### 
##############################


##############################
########## Cleaning ##########
##############################

print('Before cheating removed:', len(df_dem_formality_values))
df_dem_formality_values = df_dem_formality_values[df_dem_formality_values.apply(lambda x: x['cheating'] != 'yes', axis=1)]
print('After cheating removed:', len(df_dem_formality_values))

df_dem_formality_values = df_dem_formality_values[df_dem_formality_values.apply(lambda x: x['technical'] != 'yes', axis=1)]
print('After technical removed:', len(df_dem_formality_values))

print('Before retake and annoying education person:', len(df_dem_formality_values))
df_dem_formality_values = df_dem_formality_values[df_dem_formality_values['retake'] == '0'] 
df_dem_formality_values = df_dem_formality_values[df_dem_formality_values['education'] != 666]
print('After retake and annoying education person::', len(df_dem_formality_values))

print('Before time:', len(df_dem_formality_values))
df_dem_formality_values = df_dem_formality_values[df_dem_formality_values['time_diff'] < ((dt.timedelta(minutes=60)).total_seconds()/60)]
print('after time:', len(df_dem_formality_values))

print('Before dem drop:', len(df_dem_formality_values))
df_dem_formality_values_no_drop = df_dem_formality_values.copy()
df_dem_formality_values = df_dem_formality_values.dropna(subset=['gender', 'age', 'education'])
df_dem_formality_values = df_dem_formality_values[df_dem_formality_values['gender'] != '2']
print('after dem drop:', len(df_dem_formality_values))

print('Done cleaning formality data')
print('Total number of full participants (completed up through comments):',
      len(df_dem_formality_values))

##############################
##### Done Cleaning ##########
##############################


##############################
########## Funcs #############
##############################

# simple function for not repeating lines of code for computing and displaying linear regression results
def run_lin_regr(df, formula):
    print("Running linear regression for:", formula)
    model = ols(formula, data=df)
    results = model.fit()
    display(results.summary())

def run_mixed_regr(df, formula, g):
    print("Running mixed regression for:", formula)
    model = smf.mixedlm(formula, df, groups=g)
    
    results = model.fit()
    display(results.summary())
    
# useful for checking all values at once
value_formula = ''.join([v+' + ' for v in values]).strip(' + ')



"Everything that follow is from the Formality Regression Results file, but remodeled to fit our needs"

df_formality_value_dem_dropout = df_formality_dropout.merge(df_dem_formality_values.drop(['Turker', 'formality_level'], axis=1), on='uuid', how='left')

### Useful vars

full_base_formula = "~ C(formality_level, Treatment(reference='informal')) + C(Turker) + C(english) + C(gender) + age + education"

df_turk = df_dem_formality_values[df_dem_formality_values['Turker'] == True]
df_litw = df_dem_formality_values[df_dem_formality_values['Turker'] == False]

df_formal = df_dem_formality_values[df_dem_formality_values['formality_level'] == 'formal']
df_informal = df_dem_formality_values[df_dem_formality_values['formality_level'] == 'informal']

df_formal_turk = df_turk[df_turk['formality_level'] == 'formal']
df_informal_turk = df_turk[df_turk['formality_level'] == 'informal']

df_formal_litw = df_litw[df_litw['formality_level'] == 'formal']
df_informal_litw = df_litw[df_litw['formality_level'] == 'informal']

df_turk_dropout = df_formality_value_dem_dropout[df_formality_value_dem_dropout['Turker'] == True]
df_litw_dropout = df_formality_value_dem_dropout[df_formality_value_dem_dropout['Turker'] == False]

df_formal_dropout = df_formality_value_dem_dropout[df_formality_value_dem_dropout['formality_level'] == 'formal']
df_informal_dropout = df_formality_value_dem_dropout[df_formality_value_dem_dropout['formality_level'] == 'informal']







"ANALYSIS"


# from: https://www.danielsoper.com/statcalc/calculator.aspx?id=5
def cohen_f2(r_squared):
    return r_squared / (1 - r_squared)

def run_logit(f, df, display_summary=False):
    logitfit = smf.logit(formula = str(f), data=df, missing = 'drop').fit()
    print('---------------------------------------') 
    print(f, 'AIC:', logitfit.aic)
    if display_summary:
        display(logitfit.summary2()) 
    print('---------------------------------------') 
    return logitfit
    
def run_ols(f, df, display_summary=False):
    results = ols(f, data=df, missing = 'drop').fit()
    print('---------------------------------------') 
    print(f, 'AIC:', results.aic, 'Cohen\'s F2:', cohen_f2(results.rsquared_adj))
    if display_summary:
        display(results.summary()) 
    print('---------------------------------------') 
    return results







### RQ - Does formality influence attention?

# alternate_formula = "~ C(formality_level, Treatment(reference='informal')) + C(Turker) + C(formality_level, Treatment(reference='informal')):C(Turker)"

attention_formula = ' '.join(["engaged", full_base_formula])
model = run_logit(attention_formula, df_dem_formality_values, True)
model.summary2
### Some descriptive stats
print('--------------------------------')
print('percent engaged total:', df_dem_formality_values.engaged.sum()/len(df_dem_formality_values))
print('--------------------------------')
print('percent engaged turk:', df_turk.engaged.sum()/len(df_turk))
print('percent engaged formal turk:', df_formal_turk.engaged.sum()/len(df_formal_turk))
print('percent engaged informal turk:', df_informal_turk.engaged.sum()/len(df_informal_turk))
print('--------------------------------')
print('percent engaged litw:', df_litw.engaged.sum()/len(df_litw))
print('percent engaged formal litw:', df_formal_litw.engaged.sum()/len(df_formal_litw))
print('percent engaged informal litw:', df_informal_litw.engaged.sum()/len(df_informal_litw))
print('--------------------------------')

print('percent engaged formal:', df_formal.engaged.sum()/len(df_formal))
print('--------------------------------')
print('percent engaged informal:', df_informal.engaged.sum()/len(df_informal))
print('--------------------------------')







### RQ - Does formality influence drop out (binary)?

formula_alt = "~ C(formality_level, Treatment(reference='informal')) + C(Turker)"

dropout_count_formula = ' '.join(["dropout", formula_alt])
res = run_logit(dropout_count_formula, df_formality_value_dem_dropout, True)

print(res.params.apply(np.exp))

### Desrctiptive stats
print('checkpoint mean=', df_formality_value_dem_dropout['count'].mean(), 'std=', df_formality_value_dem_dropout['count'].std())
print('total dropout rate=', df_formality_value_dem_dropout['dropout'].sum()/len(df_formality_value_dem_dropout))








### RQ: Does formality predict time spent? 


time_formula = ' '.join(["time_diff", full_base_formula])
results = run_ols(time_formula, df_dem_formality_values, True)

### Testing specifics 
    
### Some descriptive stats
print('time total:')
print('mean=', df_dem_formality_values['time_diff'].mean(), 'std=', df_dem_formality_values['time_diff'].std())
print('--------------------------------')
print('time turk:')
print('mean=', df_turk['time_diff'].mean(), 'std=', df_turk['time_diff'].std())
print('--------------------------------')
print('time litw:')
print('mean=', df_litw['time_diff'].mean(), 'std=', df_litw['time_diff'].std())
print('--------------------------------')
print('time formal:')
print('mean=', df_formal['time_diff'].mean()/60, 'std=', df_formal['time_diff'].std()/60)
print('--------------------------------')
print('time informal:')
print('mean=', df_informal['time_diff'].mean()/60, 'std=', df_informal['time_diff'].std()/60)
# print('time formal turk:', df_formal_turk['time_diff'].sum()/len(df_formal_turk))
# print('time informal turk:', df_informal_turk['time_diff'].sum()/len(df_informal_turk))
# print('--------------------------------')
# print('time litw:', df_litw['time_diff'].sum()/len(df_litw))
# print('time formal litw:', df_formal_litw['time_diff'].sum()/len(df_formal_litw))
# print('time informal litw:', df_informal_litw['time_diff'].sum()/len(df_informal_litw))
# print('--------------------------------')

# print('time formal:', df_formal['time_diff'].sum()/len(df_formal))
# print('--------------------------------')
# print('time informal:', df_informal['time_diff'].sum()/len(df_informal))
# print('--------------------------------')







### RQ: Does formality predict score? 
import statsmodels.api as sm

score_formula = ' '.join(["score", full_base_formula, '+C(engaged)'])

f = "score ~ C(formality_level) + C(Turker) + C(english) + C(gender) + age + education +C(engaged) + C(engaged):C(Turker) "
score_model = run_ols(f, df_dem_formality_values, True)

display(sm.stats.anova_lm(score_model, type=2))
print(score_model.params)

### Some descriptive stats
print('score total:')
print('mean=', df_dem_formality_values['score'].mean(), 'std=', df_dem_formality_values['score'].std())
print('--------------------------------')
print('score turk:')
print('mean=', df_turk['score'].mean(), 'std=', df_turk['score'].std())
print('--------------------------------')
print('score litw:')
print('mean=', df_litw['score'].mean(), 'std=', df_litw['score'].std())
print('--------------------------------')
print('score formal:')
print('mean=', df_formal['score'].mean(), 'std=', df_formal['score'].std())
print('--------------------------------')
print('score informal:')
print('mean=', df_informal['score'].mean(), 'std=', df_informal['score'].std())







### Trying without english -- not doing this
df_dem_formality_values_cleaned = df_dem_formality_values[df_dem_formality_values['english'] == 1]


### Take out english
formula = "engaged ~ C(gender) + C(formality_level) + C(Turker) + education"
logitfit = smf.logit(formula = str(formula), data=df_dem_formality_values_cleaned).fit()
display(logitfit.summary2())
print('^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^')
print('---------------------------------------') 
print(formula, 'AIC:', logitfit.aic)
print('---------------------------------------')  








#### Scratch -- does formality influence commenting behavior

def check_comments(x):
    return (x['general-comments'] != '') or (x['cheating-comments'] != '') or (x['technical-comments'] != '')


df_dem_formality_values['commented'] = df_dem_formality_values.apply(check_comments, axis=1)

df_dem_formality_values.replace(to_replace={'commented' : {True: '1', False: '0'}}, inplace = True)
df_dem_formality_values['commented'] = df_dem_formality_values['commented'].apply(pd.to_numeric)


comment_formula = ' '.join(["commented", full_base_formula + " + C(engaged)"])
comment_formula = "commented ~ C(engaged) + C(Turker) + C(english)"
model = run_logit(comment_formula, df_dem_formality_values, True)
model.summary2

demographics = ['age', 'gender', 'education', 'country0']

df_dem_formality_values_no_drop['dem_missed'] = df_dem_formality_values_no_drop[demographics].apply(lambda x: any(pd.isnull(x)), axis=1)
df_dem_formality_values_no_drop.replace(to_replace={'dem_missed' : {True: '1', False: '0'}}, inplace = True)
df_dem_formality_values_no_drop['dem_missed'] = df_dem_formality_values_no_drop['dem_missed'].apply(pd.to_numeric)


dem_missed_formula = "dem_missed ~ C(engaged) + C(Turker) + C(english) "
model = run_logit(dem_missed_formula, df_dem_formality_values_no_drop, True)
model.summary2()





'''


























