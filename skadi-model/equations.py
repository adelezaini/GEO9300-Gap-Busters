#### Equations and parametrizations for the Skadi model

import numpy as np
from numpy.polynomial.polynomial import Polynomial
import pandas as pd

########## Parameters ################

epsilon_surf = 0.97  # Emissivity of the surface
sigma = 5.67e-8  # Stefan-Boltzmann constant (W/m^2 K^4)

lambda_vap = 2.5e6  # Latent heat of vaporization (J/kg)
rho_air = 1.225  # Air density (kg/m^3)
C_e = 0.0015  # Bulk transfer coefficient for latent heat

Cp = 1005  # Specific heat capacity of air at constant pressure (J/(kg*K))
Ch = 0.001  # Bulk transfer coefficient for sensible heat

# Albedo values - adjusted after seeing model results
alpha_snow = 0.7  # Albedo for snow-covered surface (average albedo of snow) (old:0.8)
alpha_veg = 0.2  # Albedo for shrubs (old: 0.15)

P = 1013.25  # Atmospheric pressure (hPa)

####### Equations and parametrizations ##################

############## ALBEDO
def surface_albedo(snow_cover): # snow cover fraction spans [0,1]
    return snow_cover * albedo_snow + (1 - snow_cover) * albedo_veg
    #snow_cover_fraction * (alpha_snow - alpha_veg) + alpha_veg #rescale the snow_fraction to [0.15-0.8]

############## ENERGY BALANCE EQUATION
# Outgoing shortwave radiation
def shortwave_out(SWin, snow_cover_fraction):
    albedo = surface_albedo(snow_cover_fraction)
    SWout = albedo * SWin
    return SWout

# Outgoing longwave radiation
def longwave_out(Ts): # Ts is in K
    LWout = epsilon_surf * sigma * (Ts)**4
    return LWout

# Sensible heat flux
def sensible_heat_flux(u, Ts, Ta):
    SH = rho_air * Cp * Ch * u * (Ts - Ta)
    return SH

# Latent heat flux
def latent_heat_flux(u, q_s, q_a):
    LE = lambda_vap * rho_air * C_e * u * (q_s - q_a)
    return LE
    
# Saturation vapor pressure
def e_sat(T):
    return 6.1094 * np.exp((17.625 * T) / (T + 243.04))

# Specific humidity
def specific_humidity(T, RH):
    esat = e_sat(T)
    return (0.622 * RH * esat) / (P - 0.378 * RH * esat)

# Surface energy balance
def energy_balance(SWin, SWout, LWin, LWout, SH, LE):
  return SWin - SWout + LWin - LWout - SH - LE
  
# Adjustment for surface relative humidity from air relative humidity
def surface_relative_humidity(RH_air, Ts, Ta, prec, snow_cover, u):
    """ Author: perplexity.ai
    Adjust air RH to estimate surface RH based on environmental factors for time series data.
    
    Parameters:
    All parameters are pandas Series or numpy arrays of equal length:
    RH_air (float): Relative humidity of the air [0-1]
    Ts (float): Surface temperature (°C)
    Ta (float): Air temperature (°C)
    prec (float): Precipitation amount
    snow_cover (float): Snow cover fraction (0-1)
    u (float): Wind speed (m/s)
    
    Returns:
    float: Estimated surface relative humidity (%)
    """
    
    RH_air_100 = RH_air * 100 # in %
    temp_factor = (Ts - Ta) * 0.5
    veg_factor = np.full_like(RH_air_100, 3)*(1-snow_cover) # 3 is assuming moderate vegetation, like shrubs)
    precip_factor = np.minimum(prec * 2, 5)  # Max 5% increase
    snow_factor = snow_cover * 10  # Max 10% increase
    wind_factor = np.where(u < 2, 0, np.where(u < 5, 1.5, 3))
    
    # Calculate adjusted RH
    RH_surface = (RH_air_100 + temp_factor + veg_factor + precip_factor + snow_factor - wind_factor)/100
    
    # Ensure RH is within valid range (0-100%)
    RH_surface = np.clip(RH_surface, 0, 1)
    
    return RH_surface
    
    
    
############## CLOUD FRACTION
def cloud_fraction(LWin):
    """ LWdown is dataframe"""
    # Resample data to daily max and min values
    daily_max = LWin.resample('D').max().interpolate()
    daily_min = LWin.resample('D').min().interpolate()
    
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
    
    LWclear_aligned = pd.Series(LWclear, index=daily_max.index).reindex(LWin.index, method='nearest')
    
    # Calculate the difference between the original data and LWclear
    difference = LWclear_aligned - LWin['LWdown']
    difference_constrained = np.clip(difference, 0, cutoff_value)

    cloud_fraction = difference_constrained/cutoff_value

    return cloud_fraction.to_frame(name="CloudFrac")
    
    
    
############## SNOW COVER FRACTION
def snow_accumulation(data):
    
    data['snow_fall'] = data['Snowf'] * 3600 #mm # Convert snowfall rate to snow cover (kg/m² to mm of snow equivalent)
    data['cumulative_snow_cover'] = data['snow_fall'].cumsum() # Calculate cumulative snow cover over time

    return data
    
def snow_melt_model(T_air, RH, rainfall, time_step_hours=1):
    """
    Calculate snow melt based on air temperature, relative humidity, and rainfall.
    
    Parameters:
    - T_air: Air temperature (K)
    - RH: Relative humidity (0-1)
    - rainfall: Rainfall rate (kg/m²/s)
    - time_step_hours: Time step of the simulation (hours)
    
    Returns:
    - total_melt: Total amount of snow melted (mm) for the given time step
    """
    # Constants
    C_m = 0.125  # Degree-hour factor (mm/°C/hour), adjusted from daily factor
    C_h = 0.1 / 24  # Humidity factor (mm/mb/hour), adjusted from daily factor
    T_b = 273.15  # Base temperature (0°C in K)

    # Convert units
    T_a = T_air - 273.15  # Convert K to °C
    P_r = rainfall * 3600 * time_step_hours  # Convert kg/m²/s to mm for the time step

    # Temperature-induced melt
    M_T = max(0, C_m * (T_a - (T_b - 273.15)) * time_step_hours)

    # Rain-induced melt
    M_R = 0.0125 * T_a * P_r

    # Humidity-induced melt
    e_s = 6.11  # Saturation vapor pressure at the snow surface (mb)
    e_a = RH * 6.11 * np.exp((17.27 * T_a) / (T_a + 237.3))  # Actual vapor pressure of the air (mb)
    M_H = C_h * (e_a - e_s) * time_step_hours

    # Total melt
    total_melt = max(0, M_T + M_R + M_H)

    return total_melt, M_T, M_R, M_H
    
def snow_cover_fraction(snow_depth, complete_cover_threshold=100):
    """
    Estimate snow cover fraction from snow depth.
    
    Parameters:
    - snow_depth: Net amount of snow on the ground (mm)
    - complete_cover_threshold: Snow depth at which the cover is considered complete (mm)
    
    Returns:
    - fraction: Estimated snow cover fraction [0-1]
    """
    # Logistic function parameters
    k = 0.05  # Steepness of the curve
    x0 = complete_cover_threshold / 2  # Midpoint of the curve
    
    # Calculate snow cover fraction using a logistic function
    fraction = 1 / (1 + np.exp(-k * (snow_depth - x0)))
    
    return fraction
    
    
    
    
########## SURFACE TEMPERATURE CONSTRAIN
"""
def T_constrain(Ts, Tair, snow_cover, prec):
  #Avoid Ts, when evaluated numerically, to be too far off compared to the Tair
  if snow_cover > 0.5:  # High snow cover: limit Ts deviation
      Ts = max(min(Ts, Tair + 2), Tair - 2)  # Allow a 2 K deviation from Tair
  elif prec > 0:  # During precipitation: keep Ts closer to Tair
      Ts = max(min(Ts, Tair + 5), Tair - 5)  # Allow a 5 K deviation from Tair
  else:  # No snow/precip: allow more variability
      Ts = max(min(Ts, Tair + 10), Tair - 10)  # Allow up to 10 K deviation
  return Ts"""
    
    

