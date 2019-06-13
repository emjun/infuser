import os
import subprocess
import runpy
import argparse
import csv # if we want to write out to an external file or something

base_path = './testprograms/'
bugs_base_path = os.path.join(base_path,'buggy_scripts/')
bugs_scripts_list_path = os.path.join(bugs_base_path, 'buggy.csv') # CSV of all the script names, the number of bugs in each, and the lines where the bugs occur

no_bugs_path = os.path.join(base_path, 'no_bugs_scripts')
no_bugs_scripts_list_path = os.path.join(no_bugs_path, 'no_bugs.csv')

def test_bugs(): 
    global bugs_scripts_list_path

    print('..................BUGGY SCRIPTS.....................')

    with open(bugs_scripts_list_path, newline='', encoding='utf-8-sig') as listfile: 
        listreader = csv.reader(listfile, delimiter=',')

        # for all the scripts named in a file and the line number that should show a bug
        for row in listreader: 
            script_name = row[0]
            num_bugs = row[1]
            bug_lines = row[2] # There may be multiple --> should break this up into list? 

            print(f"File: {script_name}\n")
            print(f"Supposed to have bugs: {num_bugs}\n")
            print(f"Lines of bugs: {bug_lines}\n")
            script_path = os.path.join(bugs_base_path, script_name)
            os.system(f"python3 -m infuser {script_path}")
            print("-------------------------------")

def test_no_bugs():
    global no_bugs_scripts_list_path 

    print('..................NO BUGS SCRIPTS.....................')

    with open(no_bugs_scripts_list_path, newline='', encoding='utf-8-sig') as listfile: 
        listreader = csv.reader(listfile, delimiter=',')

        # for all the scripts named in a file and the line number that should show a bug
        for row in listreader: 
            script_name = row[0]
            num_bugs = row[1]

            print(f"File: {script_name}\n")
            print(f"Supposed to have bugs: {num_bugs}\n")
            script_path = os.path.join(no_bugs_path, script_name)
            os.system(f"python3 -m infuser {script_path}")
            print("-------------------------------")


def main(): 
    print('Start Testing....................................')
    test_bugs()
    print('\n\n')
    test_no_bugs()
    print('End Testing....................................')

main()