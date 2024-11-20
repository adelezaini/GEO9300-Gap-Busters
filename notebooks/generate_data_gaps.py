# packages:
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gamma
import pandas as pd
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gamma

# number of file to save to:
file_number = 5


#--------------------------------------------------------------------------------
# Create power cut gaps on the synthetic data:

# first load synthetic dataset:
folder_path = '../data/synthetic_dataset'
folder_path = '/home/mlahlbac/projects/GEO9300/GEO9300-Gap-Busters/data/synthetic_dataset'
csv_files = glob.glob(os.path.join(folder_path, '*.csv'))

dfs = []
for file in csv_files:
    data = pd.read_csv(file, parse_dates=['time'], sep=',')
    dfs.append(data)

syn_ds = dfs[0]
for data in dfs[1:]:
    syn_ds = pd.merge_ordered(syn_ds, data, on='time')

print(syn_ds.head())

# Simulate power voltage -- when under threshold, station powers off.

# Station parameters:
battery_capacity_ah = 65 * 4  # Ampere per battery * no batteries
station_power_w = 50  # Station power consumption (watts) -- complete guestimate
panel_efficiency = 0.18  # Efficiency of solar panels
panel_area_m2 = 1 * 4  # Total in m² for panels
max_voltage_fully_charged = 14.36  # Upper limit for battery voltage
low_voltage_threshold = 11.19  # Voltage under which the station shuts down
max_battery_capacity_wh = battery_capacity_ah * 12  # Total energy capacity in watt-hours

solar_radiation = syn_ds['SWin']

battery_charge = max_battery_capacity_wh  # Start with full batteries charge
time_step = 1  # hourly data

battery_voltage = []

for radiation in solar_radiation:
    power_generated = radiation * panel_area_m2 * panel_efficiency 

    if power_generated >= station_power_w:
        battery_charge += (power_generated - station_power_w) * time_step  # Net gain
    else:
        battery_charge -= (station_power_w - power_generated) * time_step  # Net loss

    battery_charge = max(0, min(battery_charge, max_battery_capacity_wh)) # max value just set based on the V_batt data

    if battery_charge > 0:
        voltage = low_voltage_threshold + (battery_charge / max_battery_capacity_wh) * (max_voltage_fully_charged - low_voltage_threshold)
    else:
        voltage = np.nan  # Gap in data due to power cut!

    battery_voltage.append(voltage)

# get lengths and number of power cuts:
syn_ds['V_batt'] = battery_voltage
syn_ds['power_cuts'] = np.where(syn_ds['V_batt'].isnull(), 1, 0)
syn_ds['power_cut_length'] = syn_ds['power_cuts'].groupby((syn_ds['power_cuts'] != syn_ds['power_cuts'].shift()).cumsum()).transform('sum')
syn_ds['group'] = (syn_ds['power_cut_length'] != syn_ds['power_cut_length'].shift()).cumsum()
power_cut_length = syn_ds.groupby('group')['power_cut_length'].first().reset_index(drop=True)
power_cut_length = power_cut_length.dropna()
power_cut_length = power_cut_length.astype(int)

#--------------------------------------------------------------------------------
# Create synthetic filtered data gaps:

# first - get the gap characteristics from original data (Tuddal)
df = pd.read_csv("/home/mlahlbac/COURSES/Geophysical_data_science/Tuddal_data.csv", na_values=np.nan)

#----- Gap length calculation -----
df['filtered_LE_group'] = np.where((df['LE'].notnull()) & (df['LE_qc0'].isnull()), 1, np.nan)
df['filtered_LE_length'] = df['filtered_LE_group'].groupby((df['filtered_LE_group'] != df['filtered_LE_group'].shift()).cumsum()).transform('sum')
df['filtered_LE_length'] = np.where(df['filtered_LE_length'] == 0, np.nan, df['filtered_LE_length'])
df['group'] = (df['filtered_LE_length'] != df['filtered_LE_length'].shift()).cumsum()
filtered_LE_length = df.groupby('group')['filtered_LE_length'].first().reset_index(drop=True)
filtered_LE_length = filtered_LE_length.dropna()
filtered_LE_length = filtered_LE_length.astype(int)

print(f'Number of gaps due to filtering in original ds: {len(filtered_LE_length)}')

# apply length distribution to synthetic data

original_data_gap_lengths = filtered_LE_length
plt.hist(original_data_gap_lengths, bins=20, density=True, alpha=0.6, color='g', edgecolor='black')

# Fit a gamma distribution to the data
fit_alpha, fit_loc, fit_beta = gamma.fit(original_data_gap_lengths)
print(f"Gamma distribution parameters: alpha={fit_alpha}, loc={fit_loc}, beta={fit_beta}")

# Plot the fitted distribution
x = np.linspace(0, max(original_data_gap_lengths), 100)
pdf_fitted = gamma.pdf(x, fit_alpha, fit_loc, fit_beta)
plt.plot(x, pdf_fitted, 'r-', label='gamma pdf')

plt.xlabel('Gap Length')
plt.ylabel('Density')
plt.title('Histogram of Original Data Gaps and Fitted Gamma Distribution')
plt.legend()
plt.show()

# Parameters for gaps in data:

# no of months classified as winter vs summer:
winter_months = 5
summer_months = 7

# number of gaps per season:
summer_LE = (len(filtered_LE_length) * summer_months * 0.3)
summer_LE = int(summer_LE)
winter_LE = (len(filtered_LE_length) * winter_months * 0.8)
winter_LE = int(winter_LE)

# Generate synthetic data gaps using the fitted gamma distribution
synthetic_gap_lengths_summer = gamma.rvs(fit_alpha, fit_loc, fit_beta, size=summer_LE)
synthetic_gap_lengths_summer = np.round(synthetic_gap_lengths_summer).astype(int) # to integer

synthetic_gap_lengths_winter = gamma.rvs(fit_alpha, fit_loc, fit_beta, size=winter_LE)
synthetic_gap_lengths_winter = np.round(synthetic_gap_lengths_winter).astype(int) # to integer

# Now that we have the length of hte gaps, determine where to insert them in the syn_ds dataframe:

# Weighting so that there is a higher probability of gaps in the night time:
night_weight = 0.75
day_weight = 0.25
hours = np.arange(24)

# Create a probability distribution for the hours of the day
hour_probs = np.zeros(24)
hour_probs[0:6] = night_weight
hour_probs[6:20] = day_weight
hour_probs[20:] = night_weight

# Normalize the probabilities
hour_probs /= hour_probs.sum()

# Generate the synthetic gap locations
synthetic_gap_hour_location_summer = np.random.choice(hours, size=len(synthetic_gap_lengths_summer), p=hour_probs)
synthetic_gap_hour_location_winter = np.random.choice(hours, size=len(synthetic_gap_lengths_winter), p=hour_probs)

# Identify winter and summer periods in the dataset
def determine_season(month):
    if month in [12, 1, 2, 3, 11]:  # Define winter months
        return 'winter'
    else:
        return 'summer'

syn_ds['season'] = syn_ds['time'].dt.month.apply(determine_season)
syn_ds['hour'] = syn_ds['time'].dt.hour

# Function to insert gaps into the dataset
def insert_gaps(dataframe, gap_lengths, gap_hours, season, gap_type="LE"):
    gap_indices = []
    for length, hour in zip(gap_lengths, gap_hours):
        possible_times = dataframe[(dataframe['season'] == season) & (dataframe['hour'] == hour)].index
        if len(possible_times) == 0:
            continue
        start_idx = np.random.choice(possible_times)
        gap_indices.extend(range(start_idx, start_idx + length))
    
    gap_indices = sorted(set(gap_indices))
    gap_col_name = f'gaps_{gap_type}_{season}'
    dataframe[gap_col_name] = dataframe['time'].notna()  
    dataframe.loc[gap_indices, gap_col_name] = np.nan

    return dataframe

# Insert gaps into the dataframe
syn_ds = insert_gaps(syn_ds, synthetic_gap_lengths_summer, synthetic_gap_hour_location_summer, 'summer', gap_type="LE")
syn_ds = insert_gaps(syn_ds, synthetic_gap_lengths_winter, synthetic_gap_hour_location_winter, 'winter', gap_type="LE")

# Check how the gaps are distributed in the ds:
print(syn_ds.isna().sum())

# merge the generated NaN values for indicating missing data as a column in the synthetic daataset:
syn_ds['LE_gaps_structured'] = np.where((syn_ds['V_batt'].isnull()) | (syn_ds['gaps_LE_summer'].isnull()) | (syn_ds['gaps_LE_winter'].isnull()), np.nan, syn_ds['LE'])

# keep only time, LE and LE_gaps_structured columns:
syn_ds = syn_ds[['time', 'LE', 'LE_gaps_structured']]
#syn_ds.to_csv('../data/LE_gaps/structured_gaps_1.csv', index=False)

#--------------------------------------------------------------------------------

# RANDOM GAPS:

# Based on the LE_gaps_structured column, make the same number and length of gaps, but distribute them completely randomly over the dataset:

# When NaN, set 1 in new column, otherwise NaN.
# 1 = data missing 
syn_ds['LE_gaps_structured_boolean'] = np.where(syn_ds['LE_gaps_structured'].isnull(), 1, np.nan)
# for each group of 1s, calculate the sum of the group. Set this sum to all values in the group.
syn_ds['LE_gaps_structured_lengths'] = syn_ds['LE_gaps_structured_boolean'].groupby((syn_ds['LE_gaps_structured_boolean'] != syn_ds['LE_gaps_structured_boolean'].shift()).cumsum()).transform('sum')
 # set 0s to NaN:
syn_ds['LE_gaps_structured_lengths'] = np.where(syn_ds['LE_gaps_structured_lengths'] == 0, np.nan, syn_ds['LE_gaps_structured_lengths'])
# arrays with lengths of data gaps:
syn_ds['group'] = (syn_ds['LE_gaps_structured_lengths'] != syn_ds['LE_gaps_structured_lengths'].shift()).cumsum()
LE_gaps_structured_lengths = syn_ds.groupby('group')['LE_gaps_structured_lengths'].first().reset_index(drop=True)


# plot histogram of the lengths of the gaps:
plt.hist(LE_gaps_structured_lengths)

# number of missing values in total:
missing_values = syn_ds['LE_gaps_structured'].isnull().sum()
print(f"Total number of missing values: {missing_values}")

# Identify the first value in each group of subsequent same values
syn_ds['group'] = (syn_ds['LE_gaps_structured_lengths'] != syn_ds['LE_gaps_structured_lengths'].shift()).cumsum()
unique_gap_lengths = syn_ds.groupby('group')['LE_gaps_structured_lengths'].first().reset_index(drop=True)
unique_gap_lengths = unique_gap_lengths.dropna()
unique_gap_lengths = unique_gap_lengths.astype(int)
# sort in descending order: 
unique_gap_lengths = unique_gap_lengths.sort_values(ascending=False)

syn_ds['LE_gaps_random'] = syn_ds['LE']

def insert_random_gaps(df, gap_lengths):
    df_copy = df.copy()
    free_indices = np.where(df_copy.notna())[0]
    np.random.shuffle(free_indices)  # Shuffle initial free indices to randomize starting point

    for gap_length in gap_lengths:
        inserted = False
        attempts = 0
        while not inserted and attempts < 10000:  # Limit attempts to prevent infinite loop
            attempts += 1
            if len(free_indices) < gap_length:
                print(f"Unable to insert gap of length {gap_length}: not enough free indices.")
                break
            valid_choices = free_indices[:-gap_length + 1]
            if len(valid_choices) == 0:
                print(f"Unable to insert gap of length {gap_length}: no valid choices.")
                break
            
            start_idx = np.random.choice(valid_choices)
            gap_indices = np.arange(start_idx, start_idx + gap_length)

            if df_copy.iloc[gap_indices].notna().all():
                df_copy.iloc[gap_indices] = np.nan
                free_indices = np.setdiff1d(free_indices, gap_indices)
                inserted = True

        if not inserted:
            print(f"Failed to insert gap of length {gap_length} after {attempts} attempts.")
    return df_copy

syn_ds['LE_gaps_random'] = insert_random_gaps(syn_ds['LE_gaps_random'], unique_gap_lengths)

# Visualizing the result
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(syn_ds['time'], syn_ds['LE_gaps_structured'], label='Structured Gaps')
plt.title('Structured Gaps')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(syn_ds['time'], syn_ds['LE_gaps_random'], label='Random Gaps', color='orange')
plt.title('Random Gaps')
plt.legend()

plt.tight_layout()
plt.show()

# save to files:

# structured gaps:
ds_structured = syn_ds[['time', 'LE_gaps_structured']]
ds_structured = ds_structured.rename(columns={"LE_gaps_structured": "LE_gaps"})
ds_structured.to_csv(f'../data/LE-gaps/structured_gaps_{file_number}.csv', index=False)

# random gaps:
ds_random = syn_ds[['time', 'LE_gaps_random']]
ds_random = ds_random.rename(columns={"LE_gaps_random": "LE_gaps"})
ds_random.to_csv(f'../data/LE-gaps/random_gaps_{file_number}.csv', index=False)

