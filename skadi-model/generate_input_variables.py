# copy and paste the input variables into the synthetic dataset folder, renaming the variables
import numpy as np
from equations import shortwave_out, surface_albedo
from functions import read_and_clean_dataset, open_input_file

def main():

    path = "../data/synthetic_dataset/"

    #SWin
    SWin = read_and_clean_dataset('DIR_SWdown').rename(columns = {'DIR_SWdown': 'SWin'})
    SWin.to_csv(path + 'SWin.csv')
    print("Shortwave IN - successfully saved in data/synthetic_dataset.")
    
    snow_cover = open_input_file('snow_cover_fraction', path = path, suffix='')
    
    SWout = shortwave_out(SWin['SWin'], snow_cover['snow_cover']).rename('SWout')
    SWout.to_csv(path + 'SWout.csv')
    print("Shortwave OUT - successfully saved in data/synthetic_dataset.")
    
    albedo = surface_albedo(snow_cover['snow_cover']).rename('albedo')
    albedo.to_csv(path + 'surface_albedo.csv')
    print("Surface albedo - successfully saved in data/synthetic_dataset.")
    
    LWin = read_and_clean_dataset('LWdown').rename(columns = {'LWdown': 'LWin'})
    LWin.to_csv(path + 'LWin.csv')
    print("Longwave IN - successfully saved in data/synthetic_dataset.")
    
    Tair = read_and_clean_dataset('Tair')
    Tair.to_csv(path + 'Tair.csv')
    print("Air temperature - successfully saved in data/synthetic_dataset.")
    
    RH_air = read_and_clean_dataset('RH').rename(columns = {'RH': 'RH_air'})
    RH_air.to_csv(path + 'RH_air.csv')
    print("Air relative humidity - successfully saved in data/synthetic_dataset.")
    
    u = read_and_clean_dataset('Wind').rename(columns = {'Wind': 'u'})
    u.to_csv(path + 'wind_speed.csv')
    print("Wind speed - successfully saved in data/synthetic_dataset.")
    
    prec = read_and_clean_dataset('total_precipitation').rename(columns = {'total_precipitation': 'prec'})
    prec.to_csv(path + 'total_precipitation.csv')
    print("Total precipitation - successfully saved in data/synthetic_dataset.")
    
    pressure = read_and_clean_dataset('Psurf').rename(columns = {'PSurf': 'pressure'})
    pressure['pressure'] = pressure['pressure']/100 # Pa -> hPa
    pressure.to_csv(path + 'surface_pressure.csv')
    print("Surface pressure - successfully saved in data/synthetic_dataset.")
    
    # mean value of surface pressure : 89411.91


if __name__ == "__main__":
    main()
