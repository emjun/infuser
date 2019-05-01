import os
import csv
import pandas as pd
import numpy 
from scipy import stats

# GLOBALS
data_base_path = './testprograms/data'
df = pd.DataFrame()

def load_data(file_name):
    global data_base_path, df

    file_path = os.path.join(data_base_path, file_name)
    df = pd.read_csv(file_path)

if __name__ == '__main__':
    load_data('data_0.csv')