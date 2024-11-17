
import pandas as pd

def open_input_file(variable_name, path = '../data/input_synthetic_dataset/', suffix = '_Tuddal_2019'):
    file_path = path + variable_name + suffix + ".csv"
    return pd.read_csv(file_path, index_col="time", parse_dates=True)
    
def read_and_clean_dataset(variable_name, path = '../data/input_synthetic_dataset/', suffix = '_Tuddal_2019'):
    """Upload and clean Tuddal dataset by dropping longitude and latitude columns.
    
    Parameters:
    - variable_name (str): input variable called in the input file name
    
    Returns:
    - pd.DataFrame: Cleaned DataFrame with time and RH columns only."""

    df = pd.read_csv(path + variable_name + suffix + '.csv',parse_dates=['time'], dayfirst=True)
    df = df.drop(columns=['longitude', 'latitude'])
    df = df.reset_index(drop=True) # to be safe: reset index for the cleaned DataFrame
    
    df['time'] = pd.to_datetime(df['time'])
    df.set_index('time', inplace=True)
    
    return df
    
    
def generate_dataset(variable_list, path = '../data/input_synthetic_dataset/', suffix = '_Tuddal_2019'):
  """ Usage: data = generate_dataset(['Tair', 'RH', 'total_precipitation', 'Rainf', 'Snowf'])"""
  for i, variable in enumerate(variable_list):
      if i == 0: data = read_and_clean_dataset(variable, path, suffix)
      else: data[variable] = read_and_clean_dataset(variable, path, suffix)[variable]
  return data
    
"""
def plt_timeseries(figsize=(10, 4)):
    plt.figure(figsize=figsize)
    plt.plot(df.index, df[df.columns[2]], label=df.columns[2])  # Assuming a single variable in the CSV file
    plt.title("Time Series of LWdown_Tuddal_2019")
    plt.xlabel("Time")
    plt.ylabel(df.columns[2])
    plt.grid()
    plt.legend()
    plt.show()"""
