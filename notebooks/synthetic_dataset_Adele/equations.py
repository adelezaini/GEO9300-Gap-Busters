import numpy as np

########## Parameters ################

epsilon_surf = 0.97  # Emissivity of the surface
sigma = 5.67e-8  # Stefan-Boltzmann constant (W/m^2 K^4)

lambda_vap = 2.5e6  # Latent heat of vaporization (J/kg)
rho_air = 1.225  # Air density (kg/m^3)
C_e = 0.0015  # Bulk transfer coefficient for latent heat

Cp = 1005  # Specific heat capacity of air at constant pressure (J/(kg*K))
Ch = 0.001  # Bulk transfer coefficient for sensible heat

alpha_snow = 0.8  # Albedo for snow-covered surface
alpha_veg = 0.15  # Albedo for shrubs

P = 1013.25  # Atmospheric pressure (hPa)

####### Equations and parametrizations ##################

def surface_albedo(snow_cover_fraction): # snow cover fraction spans [0,1]
    return snow_cover_fraction * (alpha_snow - alpha_veg) + alpha_veg #rescale the snow_fraction to [0.15-0.8]

# Outgoing shortwave radiation
def SWout(SWin, snow_cover_fraction):
    albedo = surface_albedo(snow_cover_fraction)
    SWout = albedo * SWin
    return SW_out

# Outgoing longwave radiation
def LWout(Ts): # Ts is in K
    LWout = epsilon_surf * sigma * (Ts)**4
    return LWout

# Sensible heat flux
def SH(u, Ts, Ta):
    SH = rho_air * Cp * Ch * u * (Ts - Ta)
    return SH

# Latent heat flux
def LE(u, q_s, q_a):
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
    RH_air (float): Relative humidity of the air (%)
    Ts (float): Surface temperature (°C)
    Ta (float): Air temperature (°C)
    prec (float): Precipitation amount
    snow_cover (float): Snow cover fraction (0-1)
    u (float): Wind speed (m/s)
    
    Returns:
    float: Estimated surface relative humidity (%)
    """
    
    temp_factor = (Ts - Ta) * 0.5
    veg_factor = np.full_like(RH_air, 3)*(1-snow_cover) # assuming moderate vegetation, like shrubs)
    precip_factor = np.minimum(prec * 2, 5)  # Max 10% increase
    snow_factor = snow_cover * 10  # Max 10% increase
    wind_factor = np.where(u < 2, 0, np.where(u < 5, 1.5, 3))
    
    # Calculate adjusted RH
    RH_surface = RH_air + temp_factor + veg_factor + precip_factor + snow_factor - wind_factor
    
    # Ensure RH is within valid range (0-100%)
    RH_surface = np.clip(RH_surface, 0, 100)
    
    return RH_surface
