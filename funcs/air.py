# -*- coding: utf-8 -*-
"""
Created on Mon May  9 15:04:53 2022

@author: vija
"""

import numpy as np

def calc_dens(wtemp):
    dens = 999.842594 + (6.793952 * 1e-2 * wtemp) - (9.095290 * 1e-3 *wtemp**2) + (1.001685 * 1e-4 * wtemp**3) - (1.120083 * 1e-6* wtemp**4) + (6.536336 * 1e-9 * wtemp**5)
    return dens


def eddy_diffusivity(rho, depth, g, rho_0):
    nx = len(depth)
    buoy = np.ones(nx) * 7e-5
    for i in range(0, nx - 1):
        buoy[i] = np.sqrt( np.abs(rho[i+1] - rho[i]) / (depth[i+1] - depth[i]) * g/rho_0 )   
    low_values_flags = buoy < 7e-5  # Where values are low
    buoy[low_values_flags] = 7e-5
    kz = 0.00706 *( 3.8 * 1e1)**(0.56) * (buoy)**(-0.43)
    return kz

def convection_surface(v0):
    # ------------inputs--------------
    # v0: average air velocity above surface (m/s)
    # bowen: Bowen's coefficient  
    # ------------outputs-------------
    # h0: sensible heat transfer coefficient due to convection (W/(m2 K))
    
    # Bowen coefficient: ratio between sensible and latent heat exchange at the surface of a water body (check units)
    # bowen = 0.46*(w_temp-air_temp)/(w_partpress-w_satwpress)*P/760
    bowen = 0.47
    h0 = bowen*(19.0 + 0.95 * v0**2) # W/(m2 K)?? <---to be converted
    return h0

def saturated_pressure(air_temp):
    # maximum water vapor pressure (kPa) in the air at air_temp (°C)
    # Tetens equation (1930) adapted to t>0°C from Monteith and Unsworth (2008) 
    Psat_kPa = 0.61078*np.exp((17.27*air_temp)/(air_temp+237.3))
    # Conversion kPa --> Pa
    Psat = 1000*Psat_kPa
    return Psat


    
    
    
    
    