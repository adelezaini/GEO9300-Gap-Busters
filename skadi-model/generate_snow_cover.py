import pandas as pd
import numpy as np
from functions import generate_dataset
from equations import snow_accumulation, snow_melt_model, evaluate_snow_cover_fraction

def snow_melt_timeseries(input_data, decouple_components = False):

  data = pd.DataFrame(index=input_data.index)
  melt_list = []; M_T_list = []; M_R_list = []; M_H_list = []

  # Run the snow melt model for each time step
  for i in range(len(data)):
      total_melt, M_T, M_R, M_H = snow_melt_model(input_data.iloc[i]['Tair'], input_data.iloc[i]['RH'], input_data.iloc[i]['Rainf'])
      melt_list.append(total_melt)
      M_T_list.append(M_T)
      M_R_list.append(M_R)
      M_H_list.append(M_H) # useful to visualize the components

  if decouple_components:
      data['temp_melt'] = M_T_list
      data['rain_melt'] = M_R_list
      data['humidity_melt'] = M_H_list
      
  data['total_melt'] = melt_list
  data['cumulative_melt'] = data['total_melt'].cumsum()
  
  return data
  

def main():

  # Create dataframe with input variable
  data = generate_dataset(['Tair', 'RH', 'total_precipitation', 'Rainf', 'Snowf'])
  
  # estimate hourly snow fall and accumulated snow
  data['snow_fall'], data['cumulative_snow_cover'] = snow_accumulation(data['Snowf'])
  
  # Snow melt model
  data = pd.concat([data, snow_melt_timeseries(data)], axis=1)
  
  # Net snow cover (snow fall - melt)
  initial_value = 187.8086 # manually set the initial value (running the code from initial value = 0)
  net_snow_cover_list = [initial_value]

  for i in range(1,len(data)):
      net_snow_cover = data['snow_fall'].iloc[i] - data['total_melt'].iloc[i]
      net_snow_cover_acc = max(0,net_snow_cover_list[-1] + net_snow_cover)
      net_snow_cover_list.append(net_snow_cover_acc)

  data['net_snow_cover'] = net_snow_cover_list
  
  # from amount of snow cover to surface snow cover fraction
  snow_cover_fraction = evaluate_snow_cover_fraction(data['net_snow_cover'], complete_cover_threshold=100)
  
  # save the snow cover fraction into the synthetic dataset folder
  df = snow_cover_fraction.to_frame('snow_cover')
  df.to_csv("../data/synthetic_dataset/snow_cover_fraction.csv")# index=False)
  
  print("Snow cover fraction successfully saved in data/synthetic_dataset.")


if __name__ == "__main__":
    main()

