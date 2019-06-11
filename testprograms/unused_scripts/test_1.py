##### Reformatting: Melting/changing between wide/long formats

import os
import csv
import pandas as pd
import numpy 
from scipy import stats

# GLOBALS
data_base_path = './testprograms/data'
df = pd.DataFrame(index=[1,2,3,4,5])

def load_data(file_name):
    global data_base_path, df

    file_path = os.path.join(data_base_path, file_name)
    df = pd.read_csv(file_path)

def test_wrangling_data_0():
    global df 

    # Compare Forking across runs
    forking = pd.DataFrame()
    run_nums = pd.unique(df['run'])

    for num in run_nums: 
        run_df = df[df['run'] == num]
        
        # This way requires keep in mind what the ordering of times is
        forking.append(run_df['forking']) 
    
        # TODO: There could be another way would be to store run number somewhere




    
    # Compare Caching across runs


    # Compare Naive across runs


    # Does it make sense to remove instances where Caching == Forking???



        

    # Compare strategies (There's some filtering/reformatting for caching and forking)


    # Subject


    # Run


## TODO Should try to create new dataframes or interact in loop??

if __name__ == '__main__':
    load_data('timing.csv')
    test_wrangling_data_0()