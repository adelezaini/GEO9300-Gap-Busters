import pandas as pd
import numpy as np
from scipy.optimize import minimize
from equations import *

def solve_Ts(SWin, SWout, LWin, Tair, RH_air, u, prec, snow_cover, pressure = 894.1191):
    # Define the equation to minimize (energy balance should approach 0)
    def equation_system(Ts):
        LWout = longwave_out(Ts[0])
        SH = sensible_heat_flux(u, Ts[0], Tair)
        RH_s = surface_relative_humidity(RH_air, Ts[0], Tair, prec, snow_cover, u)
        q_s = specific_humidity(Ts[0], RH_s, pressure)
        q_air = specific_humidity(Tair, RH_air, pressure)
        LE = latent_heat_flux(u, q_s, q_air)
        return energy_balance(SWin, SWout, LWin, LWout, SH, LE)**2  # Squared for minimization

    # Bounds for Ts: within ±15°C of Tair
    T_constrain = 15
    Ts_bounds = [(Tair - T_constrain, Tair + T_constrain)]

    # Initial guess for Ts
    initial_guess = [Tair]

    # Minimize the energy balance function
    result = minimize(equation_system, initial_guess, bounds=Ts_bounds, method='L-BFGS-B')

    # Check if the solution converged
    if result.success:
        return result.x[0]
    else:
        raise ValueError("Solution did not converge")



def calculate_surface_temperatures(df, verbose=True):
    Ts_values = []
    non_converge_count = 0  # Initialize the counter for non-convergences

    print("Estimation of Ts in progress...")
    
    for idx, row in df.iterrows():
        try:
            Ts = solve_Ts(
                SWin=row['SWin'], 
                SWout=row['SWout'], 
                LWin=row['LWin'], 
                Tair=row['Tair'], 
                u=row['u'], 
                RH_air=row['RH_air'], 
                prec=row['prec'], 
                snow_cover=row['snow_cover'],
                pressure=row['pressure']
            )
            Ts_values.append(Ts)
        except ValueError as e:
            non_converge_count += 1  # Increment the counter
            if verbose:
                print(f"{e} at index {idx}: assigning to Ts the previous hour value ({round(Ts_values[-1], 2)})")
            Ts_values.append(Ts_values[-1])  # Assign the previous value
    
    print(f"\nNumerical method of Ts - solution did not converge {non_converge_count} times. \nTs values filled with the respectively previous value in the time series.")
    
    return pd.DataFrame({'Ts': Ts_values}, index=df.index)

def main():
    
  # Import input variables
  path = '../data/synthetic_dataset/'
  file_list = ['SWin','SWout', 'LWin', 'Tair', 'RH_air', 'wind_speed', 'total_precipitation', 'snow_cover_fraction', 'surface_pressure']
  variable_list = ['SWin','SWout', 'LWin', 'Tair', 'RH_air', 'u', 'prec', 'snow_cover', 'pressure']

  for i, variable_name in enumerate(file_list):
      if i == 0: input_data = pd.read_csv(path + variable_name + '.csv', index_col="time", parse_dates=True)
      else: input_data[variable_list[i]] = pd.read_csv(path + variable_name + '.csv', index_col="time", parse_dates=True)[variable_list[i]]
      
      
  # Numerical method to evaluate Ts
  Ts_values = calculate_surface_temperatures(input_data, verbose = False)

  
  # Evaluate and save all the output variables

  print("\nEvaluating and saving RHs, Qs, Qair, LWout, SH, LE\n")
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

  print("Output data successfully saved as individual CSV files in data/synthetic_dataset.\n")

if __name__ == "__main__":
    main()

