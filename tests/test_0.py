import os
import csv
import pandas as pd
import numpy 
from scipy import stats

# GLOBALS
data_base_path = './tests/data'
df = pd.DataFrame()

def load_data(file_name):
    global data_base_path, df

    file_path = os.path.join(data_base_path, file_name)
    df = pd.read_csv(file_path)

if __name__ == '__main__':
    load_data('data_0.csv')
    
    # This should raise an error:  
    # Wranglinge
    # NONE 
    
    # Analysis
    stats.ttest_ind(df['time'], df['accuracy'])


    # This should NOT raise an error:     
    # Wrangling
    ## There are many different ways to achieve the same kind of subsetting.
    groups = []
    conditions = ['a', 'b']
    for curr_c in conditions: 
        condition = df['condition'] == curr_c
        groups.append(df[condition])
    
    # Analysis
    stats.ttest_ind(groups[0]['time'], groups[1]['time'])
    stats.ttest_ind(groups[0]['accuracy'], groups[1]['accuracy'])

    # another way to write the same thing as above without separate wrangling steps
    stats.ttest_ind(df[df['condition'] == 'a']['time'], df[df['condition'] == 'b']['time'])



    # There could be an adversarial(?) example. 
    # This should NOT raise an error:
    df['time'] = df['accuracy']
    stats.ttest_ind(df['time'], df['accuracy'])


    # TODO What would be a subsetting error? 
    # What if we have Lukas write scripts with the same datasets as me to see if he makes any bugs? 
    


    # So maybe this example/test focuses on subsetting? 
    # Another on melting/merging/reformatting into long format? 
    # Another on simple calculation errors, etc? 
    # How do we build a compelling argument that this covers the breadth of possible unit errors or that they are the most common ones, etc.?
    
    # Scope of statistical tests: Could focus on NHST. But this might not matter?? How should we pick the tests that we test for? Based on their signatures??
