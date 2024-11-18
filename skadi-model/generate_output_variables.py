import pandas as pd
import numpy as np
from scipy.optimize import minimize
from equations import *

def solve_Ts(SWin, SWout, LWin, Tair, RH_air, u, prec, snow_cover):
    # Define the equation to minimize (energy balance should approach 0)
    def equation_system(Ts):
        LWout = longwave_out(Ts[0])
        SH = sensible_heat_flux(u, Ts[0], Tair)
        RH_s = surface_relative_humidity(RH_air, Ts[0], Tair, prec, snow_cover, u)
        q_s = specific_humidity(Ts[0], RH_s)
        q_air = specific_humidity(Ts[0], RH_air)
        LE = latent_heat_flux(u, q_s, q_air)
        return energy_balance(SWin, SWout, LWin, LWout, SH, LE)**2  # Squared for minimization

    # Bounds for Ts: within ±20°C of Tair
    Ts_bounds = [(Tair - 10, Tair + 10)]

    # Initial guess for Ts
    initial_guess = [Tair]

    # Minimize the energy balance function
    result = minimize(equation_system, initial_guess, bounds=Ts_bounds, method='L-BFGS-B')

    # Check if the solution converged
    if result.success:
        return result.x[0]
    else:
        raise ValueError("Solution did not converge")

def calculate_surface_temperatures(df, verbose = True):
    Ts_values = []
    print("Estimation of Ts in progress...")
    for idx, row in df.iterrows():
        try:
            #print(f"Processing row {idx}: Tair={row['Tair']}, SWin={row['SWin']}, SWout={row['SWout']}")
            Ts = solve_Ts(
                SWin=row['SWin'],
                SWout=row['SWout'],
                LWin=row['LWin'],
                Tair=row['Tair'],
                u=row['u'],
                RH_air=row['RH_air'],
                prec=row['prec'],
                snow_cover=row['snow_cover']
            )
            Ts_values.append(Ts)
        except ValueError as e:
            if verbose: print(f"{e} at {idx}: assigning to Ts the previous hour value({round(Ts_values[-1],2)})")
            Ts_values.append(Ts_values[-1])
    return pd.DataFrame({'Ts': Ts_values}, index=df.index)

def main():
    
  # Import input variables
  path = '../data/synthetic_dataset/'
  file_list = ['SWin','SWout', 'LWin', 'Tair', 'RH_air', 'wind_speed', 'total_precipitation', 'snow_cover_fraction']
  variable_list = ['SWin','SWout', 'LWin', 'Tair', 'RH_air', 'u', 'prec', 'snow_cover']

  for i, variable_name in enumerate(file_list):
      if i == 0: input_data = pd.read_csv(path + variable_name + '.csv', index_col="time", parse_dates=True)
      else: input_data[variable_list[i]] = pd.read_csv(path + variable_name + '.csv', index_col="time", parse_dates=True)[variable_list[i]]
      
      
  # Numerical method to evaluate Ts
  Ts_values = calculate_surface_temperatures(input_data, verbose = False)
  
  # Evaluate and save all the output variables
  output_data = pd.DataFrame(index=input_data.index)
  output_data['Ts'] = Ts_values
  output_data['RH_s'] = surface_relative_humidity(input_data['RH_air'], output_data['Ts'],
                                                  input_data['Tair'], input_data['prec'],
                                                  input_data['snow_cover'], input_data['u'])
  output_data['q_s'] = specific_humidity(output_data['Ts'], output_data['RH_s'])
  output_data['q_air'] = specific_humidity(output_data['Ts'], input_data['RH_air'])
  output_data['LWout'] = longwave_out(output_data['Ts'])
  output_data['SH'] = sensible_heat_flux(input_data['u'], output_data['Ts'], input_data['Tair'])
  output_data['LE'] = latent_heat_flux(input_data['u'], output_data['q_s'], output_data['q_air'])
  
  folder_path = "../data/synthetic_dataset/"
  variables = [var for var in output_data.columns]

  for var in variables:
    df = output_data[var].to_frame()
    #df = df.dropna()
    df.to_csv(folder_path+f"{var}.csv")

  print("Output data successfully saved as individual CSV files in data/synthetic_dataset.")

if __name__ == "__main__":
    main()

