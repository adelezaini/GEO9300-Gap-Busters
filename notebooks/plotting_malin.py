import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from sklearn.metrics import r2_score

# Define the algorithms and gap data files
algorithms = ['bart', 'neural_network', 'random_forest', 'xgboost', 'linear_regression', 'lstm']
gap_data_files = ['random_gaps_1', 'structured_gaps_1']


gap_name_map = {'random_gaps_1': 'Random gaps', 'structured_gaps_1': 'Structured gaps'}
display_algorithms = ['BART', 'NN', 'RF', 'XGBoost', 'MLR', 'LSTM']

algorithm_name_map = dict(zip(algorithms, display_algorithms))

#--------------------------------------------------------------------------------

# make a heatmap plot for the gap lengths:

# Define the bins for NaN gap lengths
bins = [0, 12, 24, 72, 168, float('inf')]
bin_labels = ['0-12', '12-24', '24-72', '72-168', '>168']

# Function to calculate consecutive NaN lengths in the LE_gaps column
def calculate_gap_lengths(gap_df):
    gap_lengths = []
    current_length = 0
    for le_gap in gap_df['LE_gaps']:
        if np.isnan(le_gap):
            current_length += 1
        else:
            if current_length > 0:
                gap_lengths.append(current_length)
                current_length = 0
    if current_length > 0:
        gap_lengths.append(current_length)
    return gap_lengths

# Function to process each result file, add time column, and calculate R² values
def add_time_and_calculate_r2(result_path, gap_data_path):
    result_df = pd.read_csv(result_path)
    gap_df = pd.read_csv(gap_data_path)
    
    # Adding the time column to the results dataframe
    result_df['Time'] = pd.to_datetime(gap_df['time'])
    
    # Calculate the gap lengths
    gap_lengths = calculate_gap_lengths(gap_df)
    
    # Create a DataFrame for gap lengths
    gap_length_df = pd.DataFrame({'Time': pd.to_datetime(gap_df['time']), 'GapLength': 0})
    i = 0
    for gap_length in gap_lengths:
        gap_length_df.loc[i:i+gap_length-1, 'GapLength'] = gap_length
        i += gap_length
    
    # Assigning bins to NaN gap lengths
    gap_length_df['GapLengthBin'] = pd.cut(gap_length_df['GapLength'], bins=bins, labels=bin_labels, right=False)
    
    # Merging gap length information
    result_df = result_df.merge(gap_length_df[['Time', 'GapLengthBin']], on='Time', how='left')
    
    # Calculate R² values for each bin
    r2_dict = {}
    for label in bin_labels:
        bin_data = result_df[result_df['GapLengthBin'] == label]
        if len(bin_data) > 0:
            r2_value = r2_score(bin_data['Actual'], bin_data['Predicted'])
        else:
            r2_value = np.nan
        r2_dict[label] = r2_value
    
    return r2_dict

# List to store R² values for plotting
r2_values = []

# Process each file and calculate R² values for each bin
for alg in algorithms:
    for gap_file in gap_data_files:
        # Construct the file paths
        result_file = f'../results/{alg}_{gap_file}_predictions.csv'
        gap_data_file = f'../data/LE-gaps/{gap_file}.csv'
        
        # Calculate R² values for each bin
        r2_dict = add_time_and_calculate_r2(result_file, gap_data_file)
        r2_values.append([algorithm_name_map[alg], gap_name_map[gap_file]] + [r2_dict[label] for label in bin_labels])

# Convert the R² values to a DataFrame for plotting
r2_df = pd.DataFrame(r2_values, columns=['Algorithm', 'GapType'] + bin_labels)

# Set the index to be a combination of Algorithm and GapType for better visualization
r2_df['Category'] = r2_df.apply(lambda row: f"{row['Algorithm']} ({row['GapType']})", axis=1)
r2_df.set_index('Category', inplace=True)

# Create the heatmap
plt.figure(figsize=(12, 8))
ax = sns.heatmap(r2_df[bin_labels], annot=True, cmap='coolwarm', cbar_kws={'label': 'R²'})
plt.title('Performance for different data gap lengths')
plt.xlabel('Missing Data Gap Length (hours)')
plt.ylabel('')
plt.tight_layout()

# Show the plot
plt.show()

#--------------------------------------------------------------------------------

# combine the boxplots into one figure with subplots.

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


# Function to process each result file and add time column
def add_time_and_calculate_difference(result_path, gap_data_path):
    result_df = pd.read_csv(result_path)
    gap_df = pd.read_csv(gap_data_path)

    # Adding the time column to the results dataframe
    result_df['Time'] = gap_df['time']

    # Calculating the difference between actual and predicted values
    result_df['Difference'] = result_df['Actual'] - result_df['Predicted']

    return result_df

# Function to process each result file, add time column, filter for specific season and time period, and calculate differences (for the second plot)
def add_time_and_calculate_difference_with_conditions(result_path, gap_data_path, season, time_period):
    result_df = pd.read_csv(result_path)
    gap_df = pd.read_csv(gap_data_path)

    # Adding the time column to the results dataframe
    result_df['Time'] = pd.to_datetime(gap_df['time'])

    # Filtering for the given season months
    result_df = result_df[result_df['Time'].dt.month.isin(season)]

    # Filtering for the given time period hours
    result_df = result_df[result_df['Time'].dt.hour.isin(time_period)]

    # Calculating the difference between actual and predicted values
    result_df['Difference'] = result_df['Actual'] - result_df['Predicted']

    return result_df

# List to store all differences for plotting (for the first plot)
all_differences_1 = []
all_algorithms_1 = []
all_gaps_1 = []

# Process each file and store the results (for the first plot)
for alg in algorithms:
    for gap_file in gap_data_files:
        # Construct the file paths
        result_file = f'../results/{alg}_{gap_file}_predictions.csv'
        gap_data_file = f'../data/LE-gaps/{gap_file}.csv'

        # Process the result file and get the dataframe with differences
        result_df = add_time_and_calculate_difference(result_file, gap_data_file)

        # Store the differences along with algorithm and gap type information
        all_differences_1.extend(result_df['Difference'].values)
        all_algorithms_1.extend([algorithm_name_map[alg]] * len(result_df))
        all_gaps_1.extend([gap_name_map[gap_file]] * len(result_df))

# Create a dataframe from all stored differences (for the first plot)
plot_df_1 = pd.DataFrame({'Algorithm': all_algorithms_1, 'GapType': all_gaps_1, 'Difference': all_differences_1})

# List to store all differences for plotting (for the second plot)
all_differences_2 = []
all_algorithms_2 = []
all_gaps_2 = []
all_conditions_2 = []

# Define the conditions for plotting (for the second plot)
conditions = [
    ('Winter Nights', [11, 12, 1, 2, 3], list(range(20, 24)) + list(range(0, 7))),
    ('Summer Days', [4, 5, 6, 7, 8, 9, 10], list(range(6, 20)))
]

# Process each file and store the results (for the second plot)
for gap_file in gap_data_files:  # Random or Structured gaps
    gap_type = gap_name_map[gap_file]
    for condition_name, season, hours in conditions:
        for alg in algorithms:
            # Construct the file paths
            result_file = f'../results/{alg}_{gap_file}_predictions.csv'
            gap_data_file = f'../data/LE-gaps/{gap_file}.csv'

            # Process the result file and get the dataframe with differences
            result_df = add_time_and_calculate_difference_with_conditions(result_file, gap_data_file, season, hours)

            # Store the differences along with algorithm, gap type, and condition information
            all_differences_2.extend(result_df['Difference'].values)
            all_algorithms_2.extend([algorithm_name_map[alg]] * len(result_df))
            all_gaps_2.extend([gap_type] * len(result_df))
            all_conditions_2.extend([condition_name] * len(result_df))

# Create a combined column for hue (for the second plot)
combined_conditions = [f'{cond} ({gap})' for cond, gap in zip(all_conditions_2, all_gaps_2)]

# Create a dataframe from all stored differences (for the second plot)
plot_df_2 = pd.DataFrame({'Algorithm': all_algorithms_2, 'GapType': all_gaps_2, 'Condition': all_conditions_2, 'Difference': all_differences_2, 'CombinedCond': combined_conditions})

# Define a custom color palette for both plots
# Random gaps: green, Structured gaps: purple, Winter: blue, Summer: red
palette_1 = {
    'Random gaps': '#229954',  # green dark
    'Structured gaps': '#7DCEA0'  # green light
}
palette_2 = {
    'Winter Nights (Random gaps)': '#1f77b4',  # Blue 
    'Winter Nights (Structured gaps)': '#aec7e8',  # light blie
    'Summer Days (Random gaps)': '#D35400',  # liorange 
    'Summer Days (Structured gaps)': '#E59866'  # Red
}

# Increase the text size
plt.rcParams.update({'font.size': 18})

# Combine two plots into one figure with two subplots
fig, axes = plt.subplots(2, 1, figsize=(14, 14), sharex=False, sharey=False)

# First Plot: Prediction Errors by Algorithm and Gap Type
sns.boxplot(data=plot_df_1, x='Algorithm', y='Difference', hue='GapType', palette=palette_1, ax=axes[0],showfliers=False)
axes[0].axhline(y=0, color='black', linestyle='--')
axes[0].set_ylabel('Difference in gap-filling LE [W/m²]', fontsize=18)
axes[0].set_xlabel('')  # Remove x-axis label
axes[0].legend(title='Data gap scenario', loc='upper left', fontsize=16, title_fontsize=16)
axes[0].tick_params(axis='x', labelsize=18)
axes[0].text(-0.1, 1.05, 'a)', transform=axes[0].transAxes, size=25, weight='bold')

# Second Plot: Prediction Errors by Algorithm, Gap Type, and Condition
sns.boxplot(data=plot_df_2, x='Algorithm', y='Difference', hue='CombinedCond', palette=palette_2, dodge=True, ax=axes[1],showfliers=False)
axes[1].axhline(y=0, color='black', linestyle='--')
axes[1].set_ylabel('Difference in gap-filling LE [W/m²]', fontsize=18)
axes[1].legend(title='Condition and Gap Type', loc='lower center', fontsize=16, title_fontsize=16)
axes[1].tick_params(axis='x', labelsize=18)
axes[1].tick_params(axis='y', labelsize=18)
axes[1].text(-0.1, 1.05, 'b)', transform=axes[1].transAxes, size=25, weight='bold')

# Adjust layout
plt.tight_layout()

# Show the combined plot
plt.show()


#--------------------------------------------------------------------------------
# R2 with the 4 categories

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import r2_score

# Function to process each result file, add time column, filter for specific season and time period, and calculate R² values
def add_time_and_calculate_r2(result_path, gap_data_path, season, time_period):
    result_df = pd.read_csv(result_path)
    gap_df = pd.read_csv(gap_data_path)
    
    # Adding the time column to the results dataframe
    result_df['Time'] = pd.to_datetime(gap_df['time'])
    
    # Filtering for the given season months
    result_df = result_df[result_df['Time'].dt.month.isin(season)]
    
    # Filtering for the given time period hours
    result_df = result_df[result_df['Time'].dt.hour.isin(time_period)]
    
    # Calculating the R² value
    if len(result_df) > 0:
        r2_value = r2_score(result_df['Actual'], result_df['Predicted'])
    else:
        r2_value = float('nan')
    
    return r2_value

# List to store R² values for plotting
all_r2_values = []
all_algorithms = []
all_gaps = []
all_conditions = []

# Define the conditions for plotting
conditions = [
    ('Winter Nights', [11, 12, 1, 2, 3], list(range(20, 24)) + list(range(0, 7))),
    ('Summer Days', [5, 6, 7, 8, 9, 10], list(range(6, 20)))
]

# Process each file and store the results
for gap_file in gap_data_files:  # Random or Structured gaps
    gap_type = gap_name_map[gap_file]
    for condition_name, season, hours in conditions:
        for alg in algorithms:
            # Construct the file paths
            result_file = f'../results/{alg}_{gap_file}_predictions.csv'
            gap_data_file = f'../data/LE-gaps/{gap_file}.csv'
            
            # Process the result file and calculate the R² value
            r2_value = add_time_and_calculate_r2(result_file, gap_data_file, season, hours)
            
            # Store the R² values along with algorithm, gap type, and condition information
            all_r2_values.append(r2_value)
            all_algorithms.append(algorithm_name_map[alg])
            all_gaps.append(gap_type)
            all_conditions.append(condition_name)

# Create a combined column for hue
combined_conditions = [f'{cond} ({gap})' for cond, gap in zip(all_conditions, all_gaps)]

# Create a dataframe from all stored R² values
plot_df = pd.DataFrame({'Algorithm': all_algorithms, 'GapType': all_gaps, 'Condition': all_conditions, 'R2': all_r2_values, 'CombinedCond': combined_conditions})

# Define a custom color palette for the combined conditions
palette = {
    'Winter Nights (Random gaps)': '#023FA5',
    'Winter Nights (Structured gaps)': '#7D87B9',
    'Summer Days (Random gaps)': '#E66C00',
    'Summer Days (Structured gaps)': '#FF9966'
}

# Create the bar plot
plt.figure(figsize=(14, 7))

# Create the barplot
sns.barplot(data=plot_df, x='Algorithm', y='R2', hue='CombinedCond', palette=palette, dodge=True)

# Customize the plot
plt.title('R² Values by Algorithm, Gap Type, and Condition')
plt.ylabel('R² value')
plt.xlabel('')  # Remove x-axis label
plt.legend(title='Condition and Gap Type', loc='upper left')
plt.tight_layout()

# Show the plot
plt.show()

#--------------------------------------------------------------------------------

# test out scatter plots


# Function to process each result file, add time column, filter for specific season and time period, and calculate differences and actual values
def add_time_and_calculate_difference_with_actual(result_path, gap_data_path, season, time_period):
    result_df = pd.read_csv(result_path)
    gap_df = pd.read_csv(gap_data_path)

    # Adding the time column to the results dataframe
    result_df['Time'] = pd.to_datetime(gap_df['time'])

    # Filtering for the given season months
    result_df = result_df[result_df['Time'].dt.month.isin(season)]

    # Filtering for the given time period hours
    result_df = result_df[result_df['Time'].dt.hour.isin(time_period)]

    # Calculating the difference between actual and predicted values
    result_df['Difference'] = result_df['Actual'] - result_df['Predicted']
    
    return result_df

# List to store all differences and actual values for plotting
all_differences = []
all_actuals = []
all_algorithms = []
all_gaps = []
all_conditions = []

# Define the conditions for plotting
conditions = [
    ('Winter Nights', [11, 12, 1, 2, 3], list(range(20, 24)) + list(range(0, 7))),
    ('Summer Days', [4, 5, 6, 7, 8, 9, 10], list(range(6, 20)))
]

# Process each file and store the results
for gap_file in gap_data_files:  # Random or Structured gaps
    gap_type = gap_name_map[gap_file]
    for condition_name, season, hours in conditions:
        for alg in algorithms:
            # Construct the file paths
            result_file = f'../results/{alg}_{gap_file}_predictions.csv'
            gap_data_file = f'../data/LE-gaps/{gap_file}.csv'

            # Process the result file and get the dataframe with differences and actual values
            result_df = add_time_and_calculate_difference_with_actual(result_file, gap_data_file, season, hours)

            # Store the differences and actual values along with algorithm, gap type, and condition information
            all_differences.extend(result_df['Difference'].values)
            all_actuals.extend(result_df['Actual'].values)
            all_algorithms.extend([algorithm_name_map[alg]] * len(result_df))
            all_gaps.extend([gap_type] * len(result_df))
            all_conditions.extend([condition_name] * len(result_df))

# Create a combined column for hue
combined_conditions = [f'{cond} ({gap})' for cond, gap in zip(all_conditions, all_gaps)]

# Create a dataframe from all stored differences and actual values
plot_df = pd.DataFrame({'Algorithm': all_algorithms, 'GapType': all_gaps, 'Condition': all_conditions, 'Difference': all_differences, 'Actual': all_actuals, 'CombinedCond': combined_conditions})

# Define a custom color palette for the combined conditions
palette = {
    'Winter Nights (Random gaps)': '#023FA5',
    'Winter Nights (Structured gaps)': '#7D87B9',
    'Summer Days (Random gaps)': '#E66C00',
    'Summer Days (Structured gaps)': '#FF9966'
}

# Combine two plots into one figure with subplots using FacetGrid
g = sns.FacetGrid(plot_df, col="Algorithm", hue="GapType", col_wrap=3, height=5, palette="muted", sharex=False, sharey=False)
g.map(sns.scatterplot, "Actual", "Difference")
g.add_legend()
for ax in g.axes.flat:
    ax.axhline(y=0, color='black', linestyle='--')
g.fig.suptitle('Difference vs Actual LE by Algorithm and Gap Type', y=1.02)
plt.tight_layout()

plt.show()

# Overlay histograms using FacetGrid
g = sns.FacetGrid(plot_df, col="Algorithm", hue="GapType", col_wrap=3, height=5, palette="muted", sharex=False, sharey=False)
g.map(sns.histplot, "Actual", multiple="stack", kde=False)
g.add_legend()
g.fig.suptitle('Histogram of Actual LE by Algorithm and Gap Type', y=1.02)
plt.tight_layout()

plt.show()

#--------------------------------------------------------------------------------
