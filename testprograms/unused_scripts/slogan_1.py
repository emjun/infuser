# Importing librararies
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


"WRANGLING"


# Set options
pd.set_option('display.max_columns', None)
pd.set_option("display.max_rows",200)
pd.set_option('display.max_colwidth', 200)


# Get datasets
df_contributors = pd.read_csv('../data/study_data/contributor_5_8_18_pre_move.csv', engine='python', sep=';')
df_contributions = pd.read_csv('../data/study_data/contribution_5_8_18_pre_move.csv', engine='python', sep=';')
df_slogans = pd.read_csv('../data/study_data/studythingy.csv', engine='python', sep=';')



### Format Data ### 

# Rename columns and drop excess index column for contributors and contributor_id and design_id for slogans

df_contributions = df_contributions.rename(columns={'created':'date_created', 'data':'comparison'})
#df_contributions.columns = ['id', 'date_created', 'status', 'comparison', 'owner_id']


df_contributors = df_contributors.rename(columns={'user_id':'index2'})
#df_contributors.columns = ['id', 'demographics', 'index2', 'motivations']

df_slogans = df_slogans.rename(columns={'data':'slogan'})
#df_slogans.columns = ['id', 'slogan', 'contributor_id', 'design_id']



df_contributors = df_contributors.drop(['index2'], axis=1)
df_slogans = df_slogans.drop(['contributor_id'], axis=1)



# Add column to slogans for study name
#df_slogans['study'] = df_slogans.design_id.apply(lambda x: "Implicit Memory" if x > 12 else ("Thinking Style" if x > 6 else "Color"))

def wrapper(x):
   return "Implicit Memory" if x > 0 else ("Thinking Stlye" if x>6 else "Color")

df_slogans['study'] = df_slogans['design_id'].apply(wrapper)

"""
# had to rewrite this to make infuser work
# Convert demographics and data columns for contributors and contributions, respectively, to dictionaries
df_contributors.demographics = df_contributors.demographics.apply(lambda x: json.loads(str.strip(x, '\'')))
df_contributors.motivations = df_contributors.motivations.apply(lambda x: json.loads(str.strip(x, '\'')))
df_contributions.comparison = df_contributions.comparison.apply(lambda x: json.loads(str.strip(x, '\'')))
"""

def json_func(x):
    return json.loads(str.strip(x,'\''))

df_contributors['demographics'] = df_contributors.demographics.apply(json_func)
df_contributors['motivations'] = df_contributors.motivations.apply(json_func)
df_contributions['comparison'] = df_contributions.comparison.apply(json_func)


# convert timestamps into datetime 
df_contributions['date_created'] = pd.to_datetime(df_contributions['date_created'])

### Clean Data ####

print(len(df_contributors))


# Get cheater, again, had to rewrite this
#df_comments_cheated = df_contributions[df_contributions['comparison'].apply(lambda x: 'comments' in x and x['cheated'] == 'True')]

def cheater_wrapper(x):
    return 'comments' in x and x['cheated'] == 'True'
df_comments_cheated = df_contributions[df_contributions['comparison'].apply(cheater_wrapper)]



# Drop particpant 10, who clearly was me who forgot to put taken test as true
df_contributors = df_contributors.drop(10)

# Clean of anyone who did not finish the test
grouped_contributions = df_contributions.groupby('owner_id')
for contr in df_contributors.iterrows():
    choices = len(df_contributions[df_contributions['owner_id'] == contr[1].id])
    try: 
        if (not contr[1].motivations) or (contr[1][1]['taken_test'] == 'True') or (choices > 50) or (contr[1].id in df_comments_cheated['owner_id'].values):
            # Drop from contributors
            df_contributors = df_contributors[df_contributors['id'] != contr[1].id]
            # Drop all contributions from that contributor
            df_contributions = df_contributions[df_contributions['owner_id'] != contr[1].id]
    except KeyError:
        continue

# Drop all contributions from that empty contributor
df_contributors = df_contributors[df_contributors['id'] != 0]
df_contributions = df_contributions[df_contributions['owner_id'] != 0]



""" again, rewrite this
# Clean of and save comments and new slogans
df_comments = df_contributions[df_contributions['comparison'].apply(lambda x: 'comments' in x)]
df_new_slogans = df_contributions[df_contributions['comparison'].apply(lambda x: 'newSlogan' in x)]
df_contributions = df_contributions[df_contributions['comparison'].apply(lambda x: 'comments' not in x and 'newSlogan' not in x)]
"""
def wrapper_1(x):
    return 'comments' in x

def wrapper_2(x):
    return 'newSlogan' in x

def wrapper_3(x):
    return 'comments' not in x and 'newSlogan' not in x

df_comments = df_contributions[df_contributions['comparison'].apply(wrapper_1)]
df_new_slogans = df_contributions[df_contributions['comparison'].apply(wrapper_2)]
df_contributions = df_contributions[df_contributions['comparison'].apply(wrapper_3)]


# Break up motivations and demographics into seperate dataframes 
# note they are still contained in df_contributors
# but are now also seperated into full dataframes, rather than a json field
df_motivations = df_contributors['motivations'].apply(pd.Series).copy()
df_demographics = df_contributors['demographics'].apply(pd.Series).copy()

# Add in ids of participants
df_motivations['id'] = df_contributors.id
df_demographics['id'] = df_contributors.id

# Should all be the same
print("Loading, formatting, and cleaning done")
print('Contributors:', len(df_contributors), 'motivations:', len(df_motivations), 'demographics:', len(df_demographics))



""" another two lines that have to be rewritten """
# clean slogan col to just hold slogan string
#df_slogans.slogan = df_slogans.slogan.apply(lambda x: json.loads(str.strip(x, '\'')))
df_slogans['slogan'] = df_slogans.slogan.apply(json_func)

#df_slogans.slogan = df_slogans.slogan.apply(lambda x: x['slogan'])
def get_slogan(x):
    return x['slogan']
df_slogans['slogan'] = df_slogans.slogan.apply(get_slogan)
# df_slogans = df_slogans.set_index('slogan', drop=False)



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

### Setting framess for slogans ###
df_slogans = df_slogans.set_index('slogan', drop=False)
set_up_framings(df_slogans, 'slogan')





"ANALYSIS"



# Calculating effect size 
# taken from https://stackoverflow.com/questions/15436702/estimate-cohens-d-for-effect-size?rq=1
def cohens_d(x, y):
#     print(x.var())
#     print(y.var())
    lx = len(x) - 1
    ly = len(y)- 1
    md = abs(x.mean() - y.mean())        ## mean difference (numerator)
    pld_var = lx * x.var() + ly * y.var()
    pld_var = pld_var/(lx + ly)
    pld_var = np.sqrt(pld_var)
    return md/pld_var


def run_anova(x, non_param):
    #counts = [x[k] for k in x.keys()]
    counts = []
    for k in x.keys():
        counts.append(x[k])
        
    if non_param: 
        f, p = stats.kruskal(*counts)
    else: 
        f, p = stats.f_oneway(*counts)
    # degrees of freedom for ANOVA
    anova_btwn = len(x) - 1
    anova_wthn = (len([val for sublist in counts for val in sublist]) - (anova_btwn + 1))
    print('F ( ', anova_btwn, ', ', anova_wthn, ') =', ('%.3f' % f), ' p =', ('%.5f' % p))



# Update: now using PAIRED sample t-test
def run_tukey(motivations_dict):
#     descriptive_stats(motivations_dict, mot)
    # Get the data into correct form for tukey tests
    # The form is all data in a single list, with corrosponding catagory labels 
    # (in this case the slogan ids) in a different list
    
    unwrapped_data = []
    k_ids = []
    i = 1
    
    for k in motivations_dict.keys():
        unwrapped_data.extend(motivations_dict[k])
        # from: https://stackoverflow.com/questions/20426313/append-the-same-value-multiple-times-to-a-list
        k_ids.extend([k] * len(motivations_dict[k]))
        #i = i + 1
        i += 1
    

    # convert to int 
    unwrapped_data = list(map(int, unwrapped_data))
    results = pairwise_tukeyhsd(unwrapped_data, k_ids, alpha=0.001)
    
    # convert tukey to df 
    # Taken from https://stackoverflow.com/questions/40516810/saving-statmodels-tukey-hsd-into-a-python-panda-dataframe
    df_tukey = pd.DataFrame(data=results._results_table.data[1:], columns=results._results_table.data[0])
    
    # get only tests where reject is true 
    df_tukey_true = df_tukey[df_tukey['reject'] == True]
    
    
    print("------------------------------------------------------")
    print('Tukey results where reject is true:')
    display(df_tukey_true.sort_values(by='meandiff'))
    print("------------------------------------------------------")
    
    
    t_test_p_vals = [] 
    for i, row in df_tukey_true.iterrows():
        # for multiple testing 
        f, p = stats.ttest_rel(motivations_dict[row.group1], motivations_dict[row.group2])
        degrees_of_freedom = (len(motivations_dict[row.group1]) - 1) + (len(motivations_dict[row.group1]) - 1)
        t_test_p_vals.append(p)
        effect_size = cohens_d(np.array(motivations_dict[row.group1]), np.array(motivations_dict[row.group2]))
        print('--------------------------------------------------------')
        print('$t_{' ,degrees_of_freedom , '} =', ('%.2f' % f), '$', 'p =', '$', ('%.3f' % p), '$','d=', '$', ('%.2f' % effect_size), '$')
        print(row.group1, ': mean =', ('%.2f' % np.mean(motivations_dict[row.group1])), 'std =', ('%.2f' % np.std(motivations_dict[row.group1])))
        print(row.group2, ': mean =', ('%.2f' % np.mean(motivations_dict[row.group2])), 'std =', ('%.2f' % np.std(motivations_dict[row.group2])))
        print('d = ', effect_size)
        print('--------------------------------------------------------')
    
    # correct for multiple tests
    r, p_vals, sidak, bonferroni = multipletests(t_test_p_vals, method='fdr_bh')
    print('Corrected p-values and rejections')
    print('--------------------------------------------------------')
    print('Reject the null? ', r)
    print('Corrected p values: ', p_vals)
    
        


### Descriptive stats
df_motivations = df_motivations.apply(pd.to_numeric)

### Descriptive Stats on Demographics and Motivations ###
print('Descriptive stats - Motivations')
print('------------------------------------------------' )
print('Mean motivations:')
print(df_motivations.mean())
print('------------------------------------------------' )
print('Std motivations:')
print(df_motivations.astype(int).std())


print('Descriptive stats - Demographics')
print('------------------------------------------------' )
print('Age')
print('------------------------------------------------' )
print('Mean age:', df_demographics['contr_age'].astype(int).mean())
print('Std age:', df_demographics['contr_age'].astype(int).std())
print('------------------------------------------------' )
print('Gender')
print('------------------------------------------------' )
print('Gender breakdown:')
print(df_demographics['gender'].value_counts())
print('------------------------------------------------' )
print('Country')
print('------------------------------------------------' )
print('Country counts:')
print(df_demographics['country'].value_counts())
print('------------------------------------------------' )
print('Education')
print('------------------------------------------------' )
print('Education levels:')
print(df_demographics['contr_edu'].value_counts())
print('------------------------------------------------' )



### Count Slogan Preferences ###

# Make new df for total count of each slogan
df_slogan_ids = df_slogans[['design_id', 'slogan', 'study', 'framing']].copy()
df_slogan_ids['count'] = 0
df_slogan_ids['index'] = df_slogan_ids['design_id']
df_slogan_ids = df_slogan_ids.drop(['design_id'], axis=1)



# Get the count of each slogan for each participant based on Copeland Counting 
# At the end of this frame you should have a dict or df that contains each slogan 
# and an associated list of the counts of how many times each participant picked that slogan


#dict_contributors = {k[1].id:[] for k in df_contributors.iterrows()}
dict_contributors = {}
for k in df_contributors.iterrows():
    dict_contributors[k[1].id] = []



# Reset total count
df_slogans['total_choices'] = 0
df_slogan_ids['count'] = 0

# Dict to hold all slogan counts, seperated by study
#slogans = {k:[] for k in df_slogans['design_id']}
slogans = {}
for k in df_slogans['design_id']:
    slogans[k] = []

#frames = {k:[] for k in df_slogans['framing']}
frames = {}
for k in df_slogans['framing']:
    frames[k] = []


# Group contributions by a single user 
grouped_contributions = df_contributions.groupby('owner_id')


def get_choice_wrapper(x):
    return x.get('choice_id')


def get_compared_wrapper(x):
    return x.get('compared_id')



# Loop through each contributor:
for contr in df_contributors.iterrows():
#     print(contr[1].id)
    if (contr[1].id == 37) or (contr[1].id == 39):
        print(contr[1])
    
    try:
        # Get the choices of that contributor
        #choices = grouped_contributions.get_group(contr[1].id).comparison.apply(lambda x: x.get('choice_id'))
        choices = grouped_contributions.get_group(contr[1].id).comparison.apply(get_choice_wrapper)
        #opp_choices = grouped_contributions.get_group(contr[1].id).comparison.apply(lambda x: x.get('compared_id'))
        opp_choices = grouped_contributions.get_group(contr[1].id).comparison.apply(get_compared_wrapper)
        
        
        # Drop any NaN or none values from both lists, convert opp_choices into ints (was list types)
        
        
        #choices = [x for x in choices if str(x) != 'nan']
        
        choices_temp = []
        for x in choices:
            if str(x) != 'nan':
                choices_temp.append(x)
        choices = choices_temp
         
        opp_choices = [x for x in opp_choices if x is not None]
        opp_choices = [x[0] for x in opp_choices]
        
        
        # Convert lists both into series 
        choices = pd.Series(choices)
        opp_choices = pd.Series(opp_choices)
        
        # Get value counts of both (count of how many times each id appears)
        choice_counts = choices.value_counts()
        opp_counts = opp_choices.value_counts()
        

                 
        # Concat counts into the same dataframe and fill any empty spots with 0
        # infuser doesn' take concat, so we work around it
        df_counts = pd.concat([choice_counts, opp_counts], axis=1)
        
        '''
        d = {'max': choice_counts, 'moritz': opp_counts}
        df_counts = pd.DataFrame(data=d, columns=[0,1])
        '''
        
               
        df_counts = df_counts.reset_index()
        
        
        
        #df_counts.columns = ['design_id','choices', 'opp_choices']
        df_counts = df_counts.rename(columns={'index':'design_id', 0:'choices', 1:'opp_choices'})
        
        
        df_counts = df_counts.fillna(0)
        
        # Get Copeland count of each slogan id, store in 'total' column
        df_counts['total'] = df_counts['choices'] - df_counts['opp_choices']
        
        dict_contributors[contr[1].id] = df_counts['choices'].values 
        
        # This is just number of times people voted for a slogan 
        # does not take into account to what it was compared to
        for i, row in df_counts.iterrows():
            past_count = df_slogan_ids.iloc[int(row.design_id - 1)]['count']
            df_slogan_ids.set_value(int(row.design_id), 'count', past_count + row.choices)
        
        # Merge with slogan dataframe to get the actual slogan (rather than just the id)
        df_slogan_choices = df_slogans[['id', 'slogan', 'design_id', 'framing']].merge(df_counts, on='design_id')
        
        # add copeland count of each slogan and frame to slogan dict
        for s in df_slogan_choices.iterrows():
            slogans[s[1].design_id].append(s[1].choices);
            frames[s[1].framing].append(s[1].choices);
        
    # Since we took out contributors who didn't finish, have to add this catch in case of a keyerror
    except KeyError:
        continue


df_dem_mot = df_motivations.merge(df_demographics[['gender', 'contr_age', 'id']], on='id')
motivations = [df_dem_mot[mot] for mot in list(df_motivations.columns.drop('id'))]





### One-way ANOVAs ###

# Do slogan preferences differ across participants
# Convert dict into list of lists
# contrib_pref = [list(dict_contributors[x]) for x in dict_contributors.keys()]

contrib_pref = []
for x in dict_contributors.keys():
    contrib_pref.append(list(dict_contributors[x]))


print('Are some slogans more preferred than others?')
run_anova(slogans, False)
run_tukey(slogans)

print('Are some frames more preferred than others?')
run_anova(frames, False)
run_tukey(frames)

for s in frames.values():
    print(len(s))








