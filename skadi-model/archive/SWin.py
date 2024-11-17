import sys
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# usage: python SWin.py <time_frequency> <latitude> <max_flux>

def estimate_swin(day_of_year, hour, latitude, max_flux):
    # Parameters to shape the curve
    peak_swin = max_flux  # Maximum SWin in W/m²
    winter_min_swin = 0  # Minimum SWin in winter

    # Generate a sharp peak in the middle of the year using a Gaussian function
    mid_year_peak = 172  # Around day 172 for peak in Northern Hemisphere summer
    width = 40  # Controls width of the peak, adjust if needed

    # Gaussian-based function for SWin with latitude adjustment
    daily_max_swin = peak_swin * np.exp(-((day_of_year - mid_year_peak) ** 2) / (2 * (width ** 2)))

    # Adjust for latitude; high latitudes will reduce the overall values
    lat_factor = np.cos(np.radians(latitude - 45))
    daily_max_swin *= lat_factor

    # Ensure daily max SWin does not fall below winter minimum
    daily_max_swin = max(daily_max_swin, winter_min_swin)

    # Add diurnal cycle
    day_length = 12 + 4 * np.sin(2 * np.pi * (day_of_year - 80) / 365)  # Approximate day length
    sunrise = 12 - day_length / 2
    sunset = 12 + day_length / 2

    if sunrise <= hour <= sunset:
        diurnal_factor = np.sin(np.pi * (hour - sunrise) / (sunset - sunrise))
        swin = daily_max_swin * diurnal_factor
    else:
        swin = 0

    return swin

def SWin(time_frequency, latitude, max_flux):
    # Set up the time series based on the given time frequency
    start_date = datetime(2024, 1, 1)
    time_delta = timedelta(minutes=int(time_frequency))
    dates = [start_date + i * time_delta for i in range(int((365 * 24 * 60) / int(time_frequency)))]  # Calculate intervals

    # Generate SWin estimates
    swin_estimates = []

    for date in dates:
        day_of_year = date.timetuple().tm_yday
        hour = date.hour + date.minute / 60
        swin_estimates.append(estimate_swin(day_of_year, hour, latitude, max_flux))

    # Add random variation to simulate fluctuations
    np.random.seed(42)  # For reproducibility
    variation = np.random.normal(0, 10, len(swin_estimates))  # Mean 0, standard deviation 10 W/m²
    swin_with_variation = np.clip(np.array(swin_estimates) + variation, 0, max_flux)

    # Create a DataFrame
    df = pd.DataFrame({'datetime': dates, 'swin': swin_with_variation})
    df.set_index('datetime', inplace=True)
    
    # Output the DataFrame as CSV
    df.to_csv('swin.csv')
    print("SWin data generated and saved as swin_output.csv")
    return df

def main(time_frequency=30, latitude=59.7463, max_flux=700):
    SWin(time_frequency, latitude, max_flux)

if __name__ == "__main__":
    # Get arguments from the command line
    time_frequency = int(sys.argv[1]) if len(sys.argv) > 1 else 30  # Default to 30 minutes
    latitude = float(sys.argv[2]) if len(sys.argv) > 2 else 59.7463  # Default to 59.7463
    max_flux = float(sys.argv[3]) if len(sys.argv) > 3 else 700  # Default to 700

    main(time_frequency, latitude, max_flux)
