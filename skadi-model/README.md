Simple surface energy balance model, dependent on snow cover - Skadi is the Norse goddess of snow :)

Developped to generate the synthetic dataset.

Input variables - time series:

- SWin: incoming short wave radiation [W/m2]
- LWin: incoming long wave radiation [W/m2]
- total_precipitation: precipitation rate [kg/m2/s]
- Rainf: rainfall rate [kg/m2/s]
- Snowf: snowfall rate [kg/m2/s]
- Tair: near surface air temperature [K]
- RHair: near surface air relative humidity [0-1]
- Qair: near surface specific humidity [kg/kg]
- Wind: wind speed [m/s]

Output variables - time series: (TBD)

- cloud_fraction:
- snow_cover_fraction:
- albedo:
- SWout:
- LWout:
- SH:
- LE:
- RHs:
- Ts:
- Qs:

Equations and parametrizations can be found in equations.py

Assumptions: (TBD)

Designed by: Malin Ahlbäck, Eivind W. Ånes, Adele Zaini
Developped by: Adele Zaini by the help of Perplexity.ai and ChatGPT
