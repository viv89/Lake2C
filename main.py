# -*- coding: utf-8 -*-
"""
Created on Mon May  9 14:39:01 2022

@author: vija
"""

import pandas as pd
from funcs.io import read_data, run_validation, read_lake_props, heatmaps, plot_profiles, modify_data
from modules.physics2b import lake


lake_name = 'feeagh'   #either greifensee or feeagh for validation and plotting

#%% Read inputs
lake_fname      = 'input/' + lake_name + '.txt'

weather_file    =  'input/IRL_Belmullet.039760_IWEC.epw' #'input/CHE_ZH_Dubendorf.AP.066099_TMYx.epw'  #
profiles_file   = 'input/demand_' + lake_name + '.xlsx' 
csv_file        = 'input/LakeEnsemblR_meteo_standard.csv'  # only for feeagh

#%% Boundary conditions

num_years = 5
 
data, info = read_data(weather_file, profiles_file, 
                       n_years = num_years, 
                       print_csv = False)
if lake_name == 'feeagh':
    data = modify_data(data, csv_file, n_years = num_years)  # only for feeagh

#%% Lake data
lake_props = read_lake_props(lake_fname)
lake_props['wtemp_init'] = 5.0        # this will likely be changed according to climate
lake_props['latitude']   = info['latitude']   # is this already inside lake_props? (see lake.txt)

#%%
params = {'nt'  : len(data.index),
          'dt'  : 3600*6,  # 1 time-step = seconds per day
          'n_years': num_years}

properties = {'g'    : 9.81,  # gravity acceleration (m/s2)
              'rho_0': 998.2, # ref water density (kg/m3)
              'cp'   : 4183,  # specific heat J/(kg K)
              'alpha': 0.08}  # diffuse reflection of solar radiation (albedo) mean annual value given by Cogley, 1979)

#%% Run lake model

lk = lake(lake_props, params, properties) # Initialize lake object
lk.run(data, False) # Run lake model without exchange
simout = lk.hist

#%%
simout.index = pd.date_range(start='1/1/2010', periods=1460, freq='6H')   
simout = simout.resample('D').mean()

#%%
# # Re-run lake model with exchange
# lk = lake(lake_props, params, properties) # Re-initialize lake object
# lk.run(data, True)
# # Save modified temperature profiles
# simout['Tw_e_mod'] = lk.hist['Tw_e']
# # simout['Tw_t_mod'] = lk.hist['Tw_t']
# simout['Tw_h_mod'] = lk.hist['Tw_h']
# # Temperature differences
# simout['delta_Tw_e'] = simout['Tw_e_mod'] - simout['Tw_e']
# # simout['delta_Tw_t'] = simout['Tw_t_mod'] - simout['Tw_t']
# simout['delta_Tw_h'] = simout['Tw_h_mod'] - simout['Tw_h']

# # min(simout['delta_Tw_t'])

#%% Validate model
vdata, errors = run_validation(lake_name, simout)

#%% Plot graphs 
plot_profiles(lake_name, vdata)  # now available for geneva, greifensee and feeagh
heatmaps(lake_name, simout, vdata)       # available for all lakes

#%%
errors_Tsurf = simout['Tw_e'].values - vdata[-2.0].values
errors_Tbott = simout['Tw_e'].values - vdata[-18.0].values
