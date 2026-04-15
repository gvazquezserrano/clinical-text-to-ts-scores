import sys
import os
sys.path.append(os.path.abspath(".."))

import psycopg2
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold

# Import necessary global variables and modules
from config.config import *


#--------------------------------------------------------------------------------#
# Auxiliary functions                                                            #
#--------------------------------------------------------------------------------#

# Set data in 'icustays' to later split it into train and test
def set_data(vitals, labs=None, notes=None, key='icustay_id', label='label_death_icu'):
    
    icustays = vitals[[key, label]].groupby([key]).first().reset_index().to_numpy()

    vitals_grouped = vitals.drop(label, axis=1).groupby([key]) if ADD_VITALS else None
    labs_grouped = labs.drop(label, axis=1).groupby([key]) if ADD_LABS else None
    notes_grouped = notes.drop(label, axis=1).groupby([key]) if ADD_NOTES else None

    return icustays, vitals_grouped, labs_grouped, notes_grouped

def windowed_range(n):
    labels = np.arange(n, dtype=np.float32) / float(n-1)
    thresholds = [(labels[i] + labels[i+1]) / 2. for i in range(n-1)]

    t_start = [0.]
    t_start.extend(thresholds)

    t_stop = thresholds
    t_stop.append(1.)
    
    for i in range(n):
        yield t_start[i], t_stop[i], i

def split_test(icustays, n_labels=2, n_folds=5, shuffle=False, random_state=None):
    y = np.zeros_like(icustays[:, 1], dtype=int)
    for start, stop, i in windowed_range(n_labels):
        y[np.where(np.logical_and(icustays[:, 1] > start, icustays[:, 1] <= stop))] = i

    for i_train, i_test in StratifiedKFold(n_splits=n_folds, shuffle=shuffle, random_state=random_state).split(icustays[:, 0], y):
        yield i_train, i_test


#--------------------------------------------------------------------------------#
# Access the data                                                                #
#--------------------------------------------------------------------------------#

# Connect to db
conn = psycopg2.connect(dbname=DBNAME, user='postgres', password='postgres')
cur = conn.cursor() 

# Read vital signs
vitals = pd.read_sql_query(f'SELECT * FROM vitals_windowed_{WINDOW_LENGTH:d}h_min{MIN_LOS_ICU:d}h;', conn)
# Read in labs values
labs = pd.read_sql_query(f'SELECT * FROM labs_windowed_{WINDOW_LENGTH:d}h_min{MIN_LOS_ICU:d}h;', conn)
# Read in notes mortality risk values
notes = pd.read_sql_query(f'SELECT * FROM notes_windowed_{WINDOW_LENGTH:d}h_min{MIN_LOS_ICU:d}h_{TEXT_ENCODER}{ABLATION_TYPE};', conn)

# Close the cursor and connection to so the server can allocate bandwidth to other requests
cur.close()
conn.close()


#--------------------------------------------------------------------------------#
# Get the icustays indexes                                                       #
#--------------------------------------------------------------------------------#

icustays, vitals_grouped, labs_grouped, notes_grouped = set_data(vitals, labs, notes, key='icustay_id', label='label_death_icu')


    
#--------------------------------------------------------------------------------#
# MIMIC-III (train and test splits)                                              #
#--------------------------------------------------------------------------------#

# Convert 'i_train_iii' and 'i_test_iii' array to a DataFrame and save it to a CSV file
if DATASET == "MIMIC_III":

    # Get i_train and i_test for a single fold
    fold_generator = split_test(icustays, n_labels=2, n_folds=5, shuffle=False, random_state=None)
    # Get i_train and i_test for the first fold (fold 0)
    i_train, i_test = next(fold_generator)    
    # Train indexes for MIMIC-III dataset (includes validation)
    df = pd.DataFrame(i_train, columns=['train_index'])
    df.to_csv(I_TRAIN_PATH, index=False)
    # Test indexes for MIMIC-III dataset (and to be used in the MIMIC-III+IV dataset as well for comparability of experiments)
    df = pd.DataFrame(i_test, columns=['test_index'])
    df.to_csv(I_TEST_PATH, index=False)



#--------------------------------------------------------------------------------#
# MIMIC-III+IV (train split; we first need the test set from MIMIC-III)          #
#--------------------------------------------------------------------------------#

# Convert 'i_train_iii_iv' array to a DataFrame and save it to a CSV file
if DATASET == "MIMIC_III_IV":

    # Get the train set by substracting the indexes of the patients in the test set
    try:
        # Access the previously created 'i_test'
        i_test_csv = pd.read_csv(I_TEST_PATH)
        # Create a numpy array
        i_test = i_test_csv.to_numpy()
        i_test_flat = i_test.flatten()
        # Create a set of indexes from i_test
        i_test_set = set(i_test_flat)
        # Create a set of all indexes in icustays
        icustays_set = set(np.arange(len(icustays)))
        # Find the indexes that are in icustays_set but not in test_old_set
        i_train_new_set = icustays_set - i_test_set
        # Convert the set back to a numpy array
        i_train = np.array(list(i_train_new_set))
        # Train indexes for MIMIC-III+IV dataset (includes validation)
        df = pd.DataFrame(i_train, columns=['train_index'])
        df.to_csv(I_TRAIN_PATH, index=False)
        
    except FileNotFoundError:
        print(f"Warning: The file '{I_TEST_PATH}' was not found.")