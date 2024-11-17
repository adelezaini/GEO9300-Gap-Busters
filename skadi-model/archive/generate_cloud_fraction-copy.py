import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial.polynomial import Polynomial

def open_file(variable_name):
    file_path = "../data/input_synthetic_dataset/"+variable_name+"_Tuddal_2019.csv"
    return pd.read_csv(file_path, index_col="time", parse_dates=True)

def main():
    
    df = open_file("LWdown")
    
    # Resample data to daily max and min values
    daily_max = df.resample('D').max().interpolate()
    daily_min = df.resample('D').min().interpolate()
    
    # Prepare the x data as day numbers for fitting
    x_data = np.arange(len(daily_max))
    
    # Fit a 5th-degree polynomial to the max and min data
    poly_max = Polynomial.fit(x_data, daily_max['LWdown'], 4)  # 4th degree polynomial for max
    poly_min = Polynomial.fit(x_data, daily_min['LWdown'], 4)  # 4th degree polynomial for min
    
    # Generate fitted values
    fitted_max = poly_max(x_data)
    fitted_min = poly_min(x_data)
    
    # Adjust the fitted curves by an offset
    fitted_max_adjusted = fitted_max + 55
    fitted_min_adjusted = fitted_min - 55
    
    # Calculate the ratio between max and min
    ratio = np.mean(fitted_max_adjusted / fitted_min_adjusted)
    
    # Create additional ratio-based curves
    LWclear = fitted_max_adjusted
    LWcloudy = fitted_max_adjusted / ratio
    
    cutoff_value = 175
    
    LWclear_aligned = pd.Series(LWclear, index=daily_max.index).reindex(df.index, method='nearest')
    
    # Calculate the difference between the original data and LWclear
    difference = LWclear_aligned - df['LWdown']
    difference_constrained = np.clip(difference, 0, cutoff_value)

    cloud_fraction = difference_constrained/cutoff_value

    cloud_fraction_df = cloud_fraction.to_frame(name="Cloud Fraction")
    
    # Export the DataFrame to a CSV file
    cloud_fraction_df.to_csv("../data/synthetic_dataset/cloud_fraction.csv")

    print("Cloud Fraction successfully saved in data/synthetic_dataset.")

if __name__ == "__main__":
    main()

