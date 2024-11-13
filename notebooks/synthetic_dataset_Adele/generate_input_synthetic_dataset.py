import xarray as xr
import pandas as pd

def main():
    # Load the dataset
    file_path = "../../data/met_analysis_1_0km_nordic_v3_yr_20161108_20240528_formatSURFEXnewTuddal.nc"
    try:
        data = xr.open_dataset(file_path)
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return

    #Path to allocate data
    folder_path = "../../data/input_synthetic_dataset/"
    # Define the subset range
    subset = data.sel(time=slice("2019-01-01", "2019-12-31"))
    
    # List of variables with a time dimension
    variables = [var for var in subset.variables if 'time' in subset[var].dims]
    
    # Save each variable as a separate CSV file
    for var in variables:
        # Convert to a DataFrame for easier CSV export
        df = subset[var].to_dataframe()
        # Drop NaN values (if any) to keep the CSV clean
        #df = df.dropna()
        # Save to CSV
        df.to_csv(folder_path+f"{var}_Tuddal_2019.csv")
    
    # Close the dataset
    data.close()
    
    print("Data successfully saved as individual CSV files in data/input_synthetic_dataset.")

if __name__ == "__main__":
    main()