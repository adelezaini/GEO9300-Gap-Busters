
import pandas as pd

def open_input_file(variable_name):
    file_path = "../../data/input_synthetic_dataset/"+variable_name+"_Tuddal_2019.csv"
    return pd.read_csv(file_path, index_col="time", parse_dates=True)
    
def read_and_clean_dataset(variable_name):
    """Upload and clean Tuddal dataset by dropping longitude and latitude columns.
    
    Parameters:
    - variable_name (str): input variable called in the input file name
    
    Returns:
    - pd.DataFrame: Cleaned DataFrame with time and RH columns only."""

    df = pd.read_csv('../../data/input_synthetic_dataset/'+variable_name+'_Tuddal_2019.csv',parse_dates=['time'], dayfirst=True)
    df = df.drop(columns=['longitude', 'latitude'])
    df = df.reset_index(drop=True) # to be safe: reset index for the cleaned DataFrame
    
    return df
    
    
def generate_dataset(variable_list):
  for i, variable in enumerate(variable_list):
      if i == 0: data = read_and_clean_dataset(variable)
      else: data[variable] = read_and_clean_dataset(variable)[variable]
          
  data['time'] = pd.to_datetime(data['time'])
  data.set_index('time', inplace=True)
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
