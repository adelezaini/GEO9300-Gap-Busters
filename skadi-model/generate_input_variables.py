import numpy as np
from equations import shortwave_out, surface_albedo
from functions import read_and_clean_dataset, open_input_file

def main():

    path = "../data/synthetic_dataset/"

    #SWin
    SWin = read_and_clean_dataset('DIR_SWdown').rename(columns = {'DIR_SWdown': 'SWin'})
    SWin.to_csv(path + 'SWin.csv')
    print("Shortwave IN - successfully saved in data/synthetic_dataset.")
    
    snow_cover = open_input_file('snow_cover_fraction', path = "../data/processing_synthetic_dataset/", suffix ='')
    snow_cover = snow_cover.drop(columns=snow_cover.columns[0]).rename(columns = {'SnowCoverFrac':'snow_cover'})
    snow_cover.to_csv(path + 'snow_cover_fraction.csv')
    print("Snow cover fraction - successfully saved in data/synthetic_dataset.")
    
    SWout = shortwave_out(SWin['SWin'], snow_cover['snow_cover']).rename('SWout')
    SWout.to_csv(path + 'SWout.csv')
    print("Shortwave OUT - successfully saved in data/synthetic_dataset.")
    
    albedo = surface_albedo(snow_cover['snow_cover'])
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
    
    cloud_fraction = open_input_file('cloud_fraction', path = "../data/processing_synthetic_dataset/", suffix ='')
    cloud_fraction = cloud_fraction.drop(columns=cloud_fraction.columns[0]).rename(columns = {'CloudFrac':'cloud_fraction'})
    cloud_fraction.to_csv(path + 'cloud_fraction.csv')
    print("Cloud fraction - successfully saved in data/synthetic_dataset.")    


if __name__ == "__main__":
    main()