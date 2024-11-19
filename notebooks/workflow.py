# Description: This script is used to create a workflow for the data analysis


import pandas as pd
import glob
import os


#-------------------------------------------------------------------------------
# Load the synthetic dataset:

folder_path = '../data/synthetic_dataset'
csv_files = glob.glob(os.path.join(folder_path, '*.csv'))

dfs = []
for file in csv_files:
    data = pd.read_csv(file, parse_dates=['time'], sep=',')
    dfs.append(data)

syn_ds = dfs[0]
for data in dfs[1:]:
    syn_ds = pd.merge_ordered(syn_ds, data, on='time')

print(syn_ds.head())

#-------------------------------------------------------------------------------
# Feauture selection:

# Get the DOY from the timestamp:
syn_ds["time"] = pd.to_datetime(syn_ds["time"])
syn_ds["doy"] = syn_ds["time"].dt.dayofyear

# Get the hour of the day from the timestamp:
syn_ds["hour"] = syn_ds["time"].dt.hour

# Define features and target variable:
features = [
    "u", 
    "LWin", 
    "SWin",
    "Tair", 
    "prec", 
    "snow_cover",
    "RH_air",
    "hour",
    "doy"
    ]

y = syn_ds["LE"]
X = syn_ds[features]

#-------------------------------------------------------------------------------

# Apply scaling here:



#-------------------------------------------------------------------------------
# Train-test split -- first import the data gaps ds:

# LE_gaps_ds = pd.read_csv('../data/synthetic_dataset_with_gaps.csv', parse_dates=['time'], sep=',')

# X_train
# Y_train

# The following test set is for where there is data gaps:
# X_test
# Y_test

#--------------------------------------------------------------------------------

# Apply machine learning model here:


