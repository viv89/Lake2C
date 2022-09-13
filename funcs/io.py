# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 15:04:16 2022

@author: vija
"""
import numpy as np
import pandas as pd
from pvlib.iotools import read_epw
from pvlib.solarposition import get_solarposition
import glob
import yaml
from yaml.loader import SafeLoader
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.pyplot import *


def read_lake_props(lake_fname):
    
    lake_data  = eval(open(lake_fname).read())
    return lake_data

def read_data(weather_file, profiles_file, n_years = 1, print_csv = False):
    
    # Initialize dataframe
    data = pd.DataFrame()
    
    # Read epw data
    weather_data = read_epw(weather_file, coerce_year = None)
    info = weather_data[1]
    
    data['month'] = weather_data[0]['month'].values
    data['day']   = weather_data[0]['day'].values
    data['hour']  = weather_data[0]['hour'].values
    
    data['Te']  = weather_data[0]['temp_air'].values
    data['RHe'] = weather_data[0]['relative_humidity'].values
    data['ghi'] = weather_data[0]['ghi'].values
    data['dni'] = weather_data[0]['dni'].values
    data['dhi'] = weather_data[0]['dhi'].values            #
    data['v_wind'] = weather_data[0]['wind_speed'].values  # m/s
    data['sky_cover'] = weather_data[0]['total_sky_cover'].values/10 # 0 = clear sky, 1 = fully covered
    data['Q_lw_sky'] = weather_data[0]['ghi_infrared'].values
           
    date_range = pd.date_range(start='1/1/2018', periods=8760, freq='H')
    data.index = date_range
    
    solardata = get_solarposition(data.index, 
                                  info['latitude'], 
                                  info['longitude'], 
                                  altitude=info['altitude'], 
                                  pressure=None, method='nrel_numpy', temperature=12)
    data['solarpos_eq'] = np.multiply(solardata['elevation'].values,data['ghi'].values)
    data_daily = data.resample('6H').mean()
    # data_daily['solarpos_eq'] = np.divide(data_daily['solarpos_eq'].values,data_daily['ghi'].values)
    # Note that GHI, DNI AND DHI are in (W/m2), energy demands are in W
    del data_daily['hour']  
    
    if print_csv == True:
        csv_print(weather_data)
    
    # Read demand for space heating, cooling (10% of the total demand within 3 km from lake perimeter)
    demand = pd.read_excel(profiles_file,sheet_name = 0, header = 0)
    data_daily['cooling_demand'] = 0*data_daily['Te'].values 
    data_daily['heating_demand'] = 0*data_daily['Te'].values
    
    data_daily = pd.concat([data_daily]*n_years)

    return data_daily, info

def modify_data(data, csv_file, n_years = 1):
    
    wdata = pd.read_csv(csv_file)
    df = pd.DataFrame(wdata)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.set_index('datetime')
    start_date = '2010-01-01 00:00:00'
    end_date = '2011-01-01 00:00:00'
    mask = (df.index >= start_date) & (df.index <= end_date)
    df = df.loc[mask]
    ddf = df.resample('6H').fillna('pad')
    ddf = ddf.iloc[:1460,:]

    ddf = pd.concat([ddf]*n_years)

    data['Te']  = ddf['Air_Temperature_celsius'].values
    data['RHe'] = ddf['Relative_Humidity_percent'].values
    data['ghi'] = ddf['Shortwave_Radiation_Downwelling_wattPerMeterSquared'].values
    data['v_wind'] = ddf['Ten_Meter_Elevation_Wind_Speed_meterPerSecond'].values
    data['Q_lw_sky'] = ddf['Longwave_Radiation_Downwelling_wattPerMeterSquared'].values
    
    return data   

def csv_print(weather_data):
    wdata = weather_data[0]
    info = weather_data[1]
    col_names = ['datetime',
                 'Ten_Meter_Elevation_Wind_Speed_meterPerSecond',
                 'Air_Temperature_celsius', 
                 'Relative_Humidity_percent',
                 'Shortwave_Radiation_Downwelling_wattPerMeterSquared',
                 'Longwave_Radiation_Downwelling_wattPerMeterSquared',
                 'Sea_Level_Barometric_Pressure_pascal',
                 'Surface_Level_Barometric_Pressure_pascal',
                 'Precipitation_millimeterPerDay', 
                 'Snowfall_millimeterPerDay']   
    ddf = pd.DataFrame(columns = col_names)
    
    ddf['datetime'] = pd.date_range(start='2010-01-01 00:00:00', periods=8760, freq='1H')
    ddf = ddf.set_index('datetime')
    ddf['Ten_Meter_Elevation_Wind_Speed_meterPerSecond'] = wdata['wind_speed'].values
    ddf['Air_Temperature_celsius'] = wdata['temp_air'].values
    ddf['Relative_Humidity_percent'] = wdata['relative_humidity'].values
    ddf['Shortwave_Radiation_Downwelling_wattPerMeterSquared'] = wdata['ghi'].values
    ddf['Longwave_Radiation_Downwelling_wattPerMeterSquared'] = wdata['ghi_infrared'].values
    ddf['Sea_Level_Barometric_Pressure_pascal'] = wdata['atmospheric_pressure'].values
    ddf['Surface_Level_Barometric_Pressure_pascal'] = wdata['atmospheric_pressure'].values
    ddf['Precipitation_millimeterPerDay'] = 0*wdata['liquid_precipitation_quantity'].values
    ddf['Snowfall_millimeterPerDay'] = 0*wdata['snow_depth'].values
    
    ddf = ddf.resample('1D').mean()
    
    # strings = weather_file.split('/')[1].split('_')
    fname = 'output/LakeEnsamblR_meteo_' + info['country'] + '_' + info['state-prov'] + '_' + info['WMO_code'] + '.csv'
    ddf.to_csv(fname, date_format='%Y-%m-%d %H:%M:%S')


def read_parameters(weather_file, hubs_file):
    
    # Read epw data
    weather_data = read_epw(weather_file, coerce_year = None)
    
    # Set location (for radiation processing) from epw
    loc_settings = {'city': weather_data[1]['city'], 
                    'lat' : weather_data[1]['latitude'], 
                    'lon' : weather_data[1]['longitude'],  
                    'alt' : weather_data[1]['altitude'], 
                    'tz'  :'Europe/Rome'}  #manually set (otherwise  weather_data[1]['TZ'])
    
    files = glob.glob('input/*.yaml')
    params = dict()
    for file in files:
    
        # Open the file and load the file
        with open(file) as f:
            fname = file.split('\\')[1].split('.')[0]
            params[fname] = yaml.load(f, Loader=SafeLoader)
    
    params['loc_settings'] = loc_settings
    
    params['branches'] = pd.read_excel(hubs_file, sheet_name = 0, 
                                       header = 0, index_col=0)
    params['nodes']    = pd.read_excel(hubs_file, sheet_name = 1, 
                                       header = 0, index_col=0)
        
    return params



def run_validation(lake_name, simout):
    
    validation_file = 'input/validation/' + lake_name +'.xlsx'
    if lake_name == 'greifensee':
        # read temperatures for validation file
        # vdata = pd.read_excel(validation_file, sheet_name = 3, header = 0, 
        #                       names = ['time','1m','15m','30m'])
        vdata = pd.read_excel(validation_file, sheet_name = 0, header = 0, 
                              index_col = 0)
    elif lake_name == 'geneva':
        vdata = pd.read_excel(validation_file, sheet_name = 4, header = 0, 
                              names = ['time','2m','20m','50m','150m'])       
    elif lake_name == 'feeagh':
        vdata = pd.read_excel(validation_file, sheet_name = 0, header = 0, 
                              index_col = 0)
    
    if lake_name == 'feeagh':
        date_range = pd.date_range(start='1/1/2019', periods=366, freq='1D')
        vdata.index = date_range   
        vdata_daily = vdata
    elif lake_name == 'greifensee':
        date_range = pd.date_range(start='1/1/2019', periods=366, freq='1D')
        vdata.index = date_range   
        vdata_daily = vdata
    else:
        date_range = pd.date_range(start='1/1/2019', periods=2921, freq='3H')
        vdata.index = date_range
        vdata_daily = vdata.resample('D').mean()
    
    vdata_daily = vdata_daily[:len(simout.index)]

    # compare outputs
    vdata_daily['Tw_e']  = simout['Tw_e'].values
    # vdata_daily['Tw_t']  = simout['Tw_t'].values
    vdata_daily['Tw_h']  = simout['Tw_h'].values
    
    # Verify output in postprocessing (should be close to zero after n_years)
    simout['deltaQ_epi'] = -(simout['Q_ev'] +  simout['Q_conv'] + simout['Q_lw'] + simout['Q_diff']) + (simout['Q_sw'] - simout['Q_sw_tr'])
    # simout['deltaQ_trn'] = simout['Q_diff_b1'] 
    #
    if lake_name == 'geneva':
        errors = {'annual_balance_epi' : simout['deltaQ_epi'].sum(), 
                  'error_surf_avg' : np.mean(vdata_daily.Tw_e.values - vdata_daily['2m'].values),
                  'error_surf_max' : np.max(vdata_daily.Tw_e.values - vdata_daily['2m'].values),
                  'error_bott_avg' : np.mean(vdata_daily.Tw_h.values - vdata_daily['150m'].values),
                  'error_bott_max' : np.min(vdata_daily.Tw_h.values - vdata_daily['150m'].values)}
    elif lake_name == 'greifensee':    
        errors = {'annual_balance_epi' : simout['deltaQ_epi'].sum(), 
                  'error_surf_min' : np.min(vdata_daily.Tw_e.values - vdata_daily[-2.0].values),
                  'error_surf_max' : np.max(vdata_daily.Tw_e.values - vdata_daily[-2.0].values),
                  'error_surf_rmse': rmse(vdata_daily.Tw_e.values,vdata_daily[-2.0].values),
                  'error_bott_min' : np.min(vdata_daily.Tw_h.values - vdata_daily[-18.0].values),
                  'error_bott_max' : np.max(vdata_daily.Tw_h.values - vdata_daily[-18.0].values),
                  'error_bott_rmse': rmse(vdata_daily.Tw_h.values,vdata_daily[-18.0].values)}
    elif lake_name == 'feeagh':
        errors = {'annual_balance_epi' : simout['deltaQ_epi'].sum(), 
                  'error_surf_min' : np.min(vdata_daily.Tw_e.values - vdata_daily[-1.0].values),
                  'error_surf_max' : np.max(vdata_daily.Tw_e.values - vdata_daily[-1.0].values),
                  'error_surf_rmse': rmse(vdata_daily.Tw_e.values,vdata_daily[-1.0].values),
                  'error_bott_min' : np.min(vdata_daily.Tw_h.values - vdata_daily[-30.0].values),
                  'error_bott_max' : np.max(vdata_daily.Tw_h.values - vdata_daily[-30.0].values),
                  'error_bott_rmse': rmse(vdata_daily.Tw_h.values,vdata_daily[-30.0].values)}

    return vdata_daily, errors

def heatmaps(lake_name, simout, vdata):
    
    # pldata = simout[['Tw_e','Tw_h','Tw_h','Tw_h','Tw_h']].transpose()
    # ax = sns.heatmap(pldata, xticklabels=30, yticklabels=False)
    # plt.title(lake_name)
    # # plt.savefig('output/figure_2c_heatmap_corr_'+lake_name+'.png')
    

    # pldata2 = vdata.transpose()
    # ax2 = sns.heatmap(pldata2, xticklabels=30, yticklabels=False)
    # plt.title(lake_name)
    # # plt.savefig('output/figure_2c_heatmap_corr_'+lake_name+'.png')
    
    plt.rcParams["font.family"] = "arial"
    fs = 11
    
    rcdata = simout[['Tw_e', 'Tw_e','Tw_e', 'Tw_e',
                     'Tw_h','Tw_h','Tw_h','Tw_h','Tw_h','Tw_h','Tw_h','Tw_h','Tw_h','Tw_h',
                     'Tw_h','Tw_h','Tw_h','Tw_h','Tw_h','Tw_h','Tw_h','Tw_h','Tw_h','Tw_h',
                     'Tw_h','Tw_h','Tw_h','Tw_h','Tw_h','Tw_h','Tw_h','Tw_h','Tw_h','Tw_h',
                     'Tw_h','Tw_h','Tw_h','Tw_h','Tw_h','Tw_h','Tw_h','Tw_h','Tw_h','Tw_h',
                     'Tw_h','Tw_h','Tw_h','Tw_h','Tw_h','Tw_h','Tw_h','Tw_h','Tw_h','Tw_h']].transpose()
    
    vdata = vdata[vdata.columns[::-1]]
    vdata = vdata.loc[:, ~vdata.columns.isin(['Tw_e', 'Tw_h'])]
    simstratdata = vdata.transpose()
    
    afont = {'fontname':'Arial'}

    f,(ax1,ax2) = plt.subplots(2,1, sharey=True, sharex=True)
    
    g1 = sns.heatmap(rcdata,cbar=True,ax=ax1, vmin=0, vmax=19, 
                     cbar_kws={'label': 'Temperature (°C)'})
    g1.set_ylabel('Depth (m)', fontsize=fs)
    g1.set_xlabel('')
    g1.set_xticks([])
    ax1.set_title('2C model', fontsize=fs)

    g2 = sns.heatmap(simstratdata,xticklabels=30, cbar=True,ax=ax2, vmin=0, vmax=19, 
                     cbar_kws={'label': 'Temperature (°C)'})
    g2.set_ylabel('Depth (m)', fontsize=fs)
    g2.set_xlabel('Day of the year', fontsize=fs)
    g2.set_xticks(np.arange(0,365,30))
    ax2.set_title('Simstrat', fontsize=fs)
    
    days_ticks = np.arange(len(vdata), step = 30) + 1
    ax2.set_xticks(days_ticks)
    days_labels = days_ticks.tolist()
    ax2.set_xticklabels(days_labels)
    
    if lake_name == 'greifensee':
        f.suptitle('(b)', fontsize=12)
    elif lake_name == 'feeagh':    
        f.suptitle('(d)', fontsize=12)
    # plt.title('d')
    
    plt.savefig('output/heatmaps_'+lake_name+'.pdf')
    
    return


def plot_profiles(lake_name, vdata):
    
    afont = {'fontname':'Arial'}
    plt.rcParams["font.family"] = "arial"
    fs = 11
    
    fig, ax = plt.subplots()
    ax.plot(vdata.index, vdata['Tw_e'], 'r', linewidth=1.0, label = '2C model (surface layer)')
    # ax.plot(vdata.index, vdata['Tw_t'], 'g', linewidth=1.0, label = 'Model transition layer')
    ax.plot(vdata.index, vdata['Tw_h'], 'b', linewidth=1.0, label = '2C model (bottom layer)')
    if lake_name == 'geneva':
        ax.plot(vdata.index, vdata['2m'], 'r:', linewidth=1.5, label = 'Measured 2m')
        ax.plot(vdata.index, vdata['20m'], 'g:', linewidth=1.5, label = 'Measured 20m')
        ax.plot(vdata.index, vdata['50m'], 'b:', linewidth=1.5, label = 'Measured 50m')
    elif lake_name == 'greifensee':
        # ax.plot(vdata.index, vdata['1m'], 'r:', linewidth=1.5, label = 'Measured 1m')
        # ax.plot(vdata.index, vdata['15m'], 'g:', linewidth=1.5, label = 'Measured 15m')
        # ax.plot(vdata.index, vdata['30m'], 'b:', linewidth=1.5, label = 'Measured 30m')
        
        ax.plot(vdata.index, vdata[-2.0], 'r:', linewidth=1.5, label = 'Simstrat (z = 2 m)')
        ax.plot(vdata.index, vdata[-18.0], 'b:', linewidth=1.5, label = 'Simstrat (z = 18 m)')
        plt.title('(a)')
        
    elif lake_name == 'feeagh':
        ax.plot(vdata.index, vdata[-2.0], 'r:', linewidth=1.5, label = 'Simstrat (z = 2 m)')
        ax.plot(vdata.index, vdata[-24.0], 'b:', linewidth=1.5, label = 'Simstrat (z = 24 m)')
        plt.title('(c)', fontsize=12)
        # ax.plot(vdata.index, vdata[-30.0], 'b:', linewidth=1.5, label = 'Simstrat 30m')
    # ax.plot(vdata.index, -simout['ghi']/20, 'k', linewidth=1.0, label = 'ghi')
    # ax.plot(vdata.index, simout['Tw_e']-simout['Te'], 'b', linewidth=1.0, label = 'Te')
    
    # Major ticks every two months
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1,3, 5, 7, 9, 11)))
    # ax.xaxis.set_minor_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m'))
    
    ax.legend(frameon=False)
    ax.set_ylabel('Temperature (°C)', fontsize=fs)
    ax.set_xlabel('Date', fontsize=fs)
    plt.grid(True,linestyle='--')
    # plt.show()
    
    plt.savefig('output/profiles_'+lake_name+'.pdf')
    return

def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())
    
    


    