# -*- coding: utf-8 -*-
"""
Created on Tue May 10 11:56:22 2022

@author: vija
"""

import numpy as np
import pandas as pd

# disable warnings "A value is trying to be set on a copy of a slice from a DataFrame"
pd.options.mode.chained_assignment = None  # default='warn'

class obj(object):
    '''
        A small class which can have attributes set
    '''
    pass

# Lake module
    
class lake:
    
    def __init__(self, lake_data, params, properties):
        
        
        # initialize objects
        self.data    = obj()    
        self.params  = obj()
        self.props   = obj()
        self.vars    = obj()
        self.vardata = obj()
        self.hist    = obj()
        
        # read data and parameters
        self.data.V         = lake_data['volume']
        self.data.L         = lake_data['length']
        self.data.W         = lake_data['width']
        self.data.A_surf    = lake_data['surf_area']
        # self.data.h         = lake_data['depth']  
        # self.data.ext_coeff = lake_data['extinction_coeff']
        self.data.Tw_init   = lake_data['wtemp_init']
        self.data.latitude  = lake_data['latitude']
        
        # self.params.nz     = params['nz']
        self.params.nt     = params['nt']
        self.params.dt     = params['dt']
        
        self.props.rho_0 = properties['rho_0']
        self.props.g     = properties['g']
        self.props.cp    = properties['cp']
        # self.props.eps_w = properties['eps_w']
        self.props.alpha = properties['alpha']
        
        # Lake geometry assumption
        # self.data.h      = 3*self.data.V/(self.data.A_surf) # maximum depth of the lake (assuming conic shape)       
        # self.data.z_therm = self.thermocline_depth(self.data.L,self.data.W) # thermocline depth        
        # self.data.A_therm = self.data.A_surf*(self.data.h - self.data.z_therm)/self.data.h # thermocline area        
        # self.data.V_hypo = 1/3*self.data.A_therm*(self.data.h - self.data.z_therm) # water volume of the hypolymnion
        # self.data.V_epi  = self.data.V - self.data.V_hypo  # water volume of the epilymnion
        
        # Lake geometry assumption (changed)
        self.data.z_therm = self.thermocline_depth(self.data.L,self.data.W) # thermocline depth  
        self.data.V_epi = self.data.A_surf*self.data.z_therm 
        self.data.V_hypo = self.data.V - self.data.V_epi
        self.data.delta_z = 3*self.data.V_hypo/(2*self.data.A_surf)
        self.data.h = self.data.z_therm + self.data.delta_z
        self.data.A_therm = self.data.A_surf
        
        # self.data.dz          = self.data.h/self.params.nz       
        # self.data.depth_range = np.linspace(0,self.params.nz,self.params.nz)
        # self.data.area_range  = np.linspace(self.data.A_surf,0,self.params.nz)
        
        self.data.z_e     = self.data.z_therm/2  #average depth of the epi
        self.data.z_h     = self.data.z_therm + (self.data.h-self.data.z_therm)/2 #average depth of the hypo
        self.data.z_range = np.array((self.data.z_e,self.data.z_h))
        
        # light transmission through water column
        self.data.light_fraction = 0.02 #self.light_extinction(self.data.ext_coeff, self.data.z_therm)
        
        # initialize model variables
        self.vars.Tw  = self.data.Tw_init*np.ones(2)
        self.vars.rho = np.zeros(2)
        self.vars.kz  = np.zeros(1) 
        
        self.hist = pd.DataFrame(columns=['Te','RHe','v_wind','ghi',
                                          'Tw_e','Tw_h','Tw_avg','Psat','rho_e','rho_h','kz',
                                          'Q_ev','Q_conv','Q_lw','Q_sw','Q_sw_tr','Q_diff','Tsky'])
        self.hist.Tw_e  = np.zeros(self.params.nt)
        self.hist.Tw_h  = np.zeros(self.params.nt)
        self.hist.rho_e = np.zeros(self.params.nt)
        self.hist.rho_h = np.zeros(self.params.nt)
        self.hist.kz    = np.zeros(self.params.nt)
        self.hist.Tw_avg = np.zeros(self.params.nt)
    
    
    def run(self, vardata):
        
        # Rename variables
        Tw = self.vars.Tw
        rho = self.vars.rho
        kz  = self.vars.kz
        dt    = self.params.dt
        # dz    = self.data.z_range[1] - self.data.z_range[0]
        # eps_w   = self.props.eps_w
        alpha = self.props.alpha
        cp    = self.props.cp
        
        # Read boundary conditions
        self.vardata.Te     = vardata['Te']    # dry-bulb temperature of the outdoor air
        self.vardata.RHe    = vardata['RHe']    # partial water vapor pressure in the air
        self.vardata.v_wind = vardata['v_wind']
        self.vardata.ghi    = vardata['ghi']
        self.vardata.sc     = vardata['sky_cover']
        
        # Save boundary conditions into hist
        self.hist.Te  = vardata['Te'].values 
        self.hist.RHe  = vardata['RHe'].values 
        self.hist.v_wind = vardata['v_wind'].values 
        self.hist.ghi    = vardata['ghi'].values 
        self.hist.Q_ev   = np.zeros(self.params.nt)
        self.hist.Q_conv = np.zeros(self.params.nt)
        self.hist.Q_lw   = np.zeros(self.params.nt)
        self.hist.Q_sw   = np.zeros(self.params.nt)
        self.hist.Q_sw_tr = np.zeros(self.params.nt)
        self.hist.Q_diff  = np.zeros(self.params.nt)
        self.hist.Tsky  = np.zeros(self.params.nt)
        
        sigma = 5.67*1e-8 # W/(m2 K4) Stephan-Boltzmann constant for blackbody radiation
        # L_w   = 2260000 # J/kg Latent heat of vaporization of water
        Tw_new = np.zeros(2)
        
        # Simulation
        for t in range(self.params.nt):
            # Shorten variable names
            Te = self.vardata.Te[t]
            v  = self.vardata.v_wind[t] #m/s
            RHe = self.vardata.RHe[t]
            # calculate density and turbulent diffusivity
            rho = self.calc_dens(Tw)
            kz  = self.eddy_diffusivity(v, rho) 
            T_sky  = self.sky_temperature(Te)
            Psat   = self.saturated_pressure(Tw[0])            
            Pe     = self.saturated_pressure(Te)*RHe/100
            # heat balance of the surface layer (epilymnion)
            # Latent heat flux (W/m2) due to evaporation of surface water (Ryan, 1974)
            Twv = Tw[0]/(1-0.378*Psat/101325)
            Tav = Te/(1-0.378*Pe/101325)
            if Twv>Tav:                
                Q_ev = (0.027*(Twv-Tav)**(0.333)+0.032*v)*(Psat-Pe)  
            else:
                Q_ev = (0.032*v)*(Psat-Pe)
            # Bowen coefficient: ratio between sensible and latent heat exchange at the surface of a water body
            Q_conv = self.bowen()*Q_ev  # W/m2
            # Q_lw_out = eps_w*sigma*(Tw[0]+273.15)**4 # W/m2
            # # emissivity of the sky
            # eps_a = 0.919*1e-5*(Te+273)**2
            # Q_lw_in = eps_a*sigma*(Te+273)**2
            # Q_lw = Q_lw_out - Q_lw_in 
            # Q_lw = sigma*((Te+273.15)**4)*(0.56-0.008*Pe**0.5)*(0.1+0.9*(1-self.vardata.sc[t]))
            Q_lw = 0.9*sigma*((Tw[0]+273.15)**4-(T_sky+273.15)**4)           
            Q_sw    = (1-alpha)*self.vardata.ghi[t]      # W/m2
            Q_sw_tr = self.data.light_fraction*Q_sw  # W/m2
            # mixing between layers
            Q_diff  = kz*(Tw[0]-Tw[1])  # W/m2
            Tw_new[0] = Tw[0] + (dt*self.data.A_surf)/(rho[0]*cp*self.data.V_epi)*(-Q_ev-Q_conv-Q_lw+Q_sw-Q_diff-Q_sw_tr)
            # heat balance of the bottom layer (hypolymnion)
            Tw_new[1] = Tw[1] + (dt*self.data.A_therm)/(rho[1]*cp*self.data.V_hypo)*(Q_diff+Q_sw_tr)  
            #update temperatures
            Tw[0] = Tw_new[0]
            Tw[1] = Tw_new[1]
            # save temperature, density and turbulent diffusivity
            self.hist.Tw_e[t]  = Tw[0]
            self.hist.Tw_h[t]  = Tw[1]
            self.hist.Psat[t]  = Psat
            self.hist.rho_e[t] = rho[0]
            self.hist.rho_h[t] = rho[1]
            self.hist.kz[t]    = kz
            self.hist.Q_ev[t]   = Q_ev
            self.hist.Q_conv[t] = Q_conv
            self.hist.Q_lw[t]   = Q_lw
            self.hist.Q_sw[t]   = Q_sw
            self.hist.Q_sw_tr[t] = Q_sw_tr
            self.hist.Q_diff[t]  = Q_diff
            self.hist.Tsky[t]  = T_sky
        
        self.hist.Tw_avg = (self.data.V_epi*self.hist.Tw_e+self.data.V_hypo*self.hist.Tw_h)/(self.data.V_epi+self.data.V_hypo)
        self.hist.Tw_avg = (self.hist.Tw_e+self.hist.Tw_h)/2
            
       

    
    #--------------------------------------------------------------------------    
    # used inside this class -------------------------------------------------------  
    
    def bowen(self):
        # Woolway et al (2018) Geographic and temporal variations in turbulent 
        # heat loss from lakes: a global analysis across 45 lakes
        #
        # data from summer (jul-Sept) 
        # in winter B is higher 
        #
        lat = self.data.latitude
        B   = 0.0501*np.exp(0.0295*lat)
        # B = 0.44
        return B
        
    def thermocline_depth(self, L, W):
        #
        # Hanna M. (1990): Evaluation of Models predicting Mixing Depth. 
        # Can. J. Fish. Aquat. Sci. 47: 940-947
        #
        MEL = max(L,W)/1000
        z_therm = 10**(0.336*np.log10(MEL-0.245))  # m
        z_therm = 2*z_therm
        return z_therm
    
    def light_extinction(self, ext_coeff, z):
        # z_a = 0.6 # m
        light_fraction = 1 * np.exp(-ext_coeff*(z))
        return light_fraction
    
    def calc_dens(self, wtemp):
        dens = 999.842594 + (6.793952 * 1e-2 * wtemp) - (9.095290 * 1e-3 *wtemp**2) + (1.001685 * 1e-4 * wtemp**3) - (1.120083 * 1e-6* wtemp**4) + (6.536336 * 1e-9 * wtemp**5)
        return dens
    
    def sky_temperature(self, Te):
        # conversion °C --> K
        Te = Te + 273.15 
        # Swinbank model (1963): valid for clear sky (overestimation of Q_lw = 85 W/m2)        
        # Tsky = 0.0553*Te**1.5
        # Fuentes model (1987): assumes average cloudiness factor of 0.61 (Q_lw = 70 W/m2)        
        Tsky = 0.037536*Te**1.5 + 0.32*Te
        # conversion K --> °C
        Tsky = Tsky - 273.15
        return Tsky
    
    #--------------------------------------------------------------------------
    # used outside this class ------------------------------------------------------
    




    def eddy_diffusivity(self, v, rho):
        # 
        # Lake Water Temperature Simulation Model, Hondzo & Stefan (1987)
        #
        # # Lake surface area [km^2]
        # A_s = self.data.A_surf*1e-6  # conversion [m^2] --> [km^2]
        # # stability frequency
        # N2  = -(rho[0]-rho[1])/(self.data.z_e - self.data.z_h)*self.props.g/rho_0  
        # if N2 <= 0:
        #     N2 = 7e-5 #sec^(-2)
        # # Hypolimnetic eddy diffusivity [cm2/s]
        # kz = 8.17*1e-4*(A_s**0.56)*(N2**-0.43)
        # # Conversion 
        # kz = kz*1e-4 # [cm2/s] --> [m2/s]
        
        # rho_0 = self.props.rho_0
        # g     = self.props.g
        # # nz    = self.params.nz
        # nz = 2
        # depth = self.data.z_range
        # buoy = np.ones(nz) * 7e-5
        # for i in range(0, nz - 1):
        #     buoy[i] = np.sqrt( np.abs(rho[i+1] - rho[i]) / (depth[i+1] - depth[i]) * g/rho_0 )   
        # low_values_flags = buoy < 7e-5  # Where values are low
        # buoy[low_values_flags] = 7e-5
        # kz = 0.00706 *( 3.8 * 1e1)**(0.56) * (buoy)**(-0.43)
        
        # TwoLayer <- function(t, y, parms)-------------------------------
        Ht = 3 * 100 # thermocline thickness (cm)
        a = 7 # constant

        # # diffusion coefficient
        Cd = 0.00052*v**(0.44)        # unit ok because v is in m/s
        shear = 1.164/1000*Cd*v**2    # unit ok (see above)
        c = 9e4                       # empirical constant
        w0 = np.sqrt(shear/(rho[0]/1000)) # rho divided by 1000 because must be in g/cm3
        E0  = c * w0
        Ri = ((self.props.g/self.props.rho_0)*(abs(rho[0]-rho[1])/10))/(w0/(self.data.z_therm)**2)
        if (rho[0] > rho[1]):
            dV = 100
        else:
            dV = (E0 / (1 + a*Ri)**(3/2))/(Ht/100) #* (86400/10000)  #m2/s
            
        kz = dV  # m2/s (probabilm lui divide per 10000 per avere cm2/s 
                 #       e moltiplica per 86400 per passare da W a J/d)
        
        return kz


    def saturated_pressure(self, air_temp):
        # maximum water vapor pressure (kPa) in the air at air_temp (°C)
        # Tetens equation (1930) adapted to t>0°C from Monteith and Unsworth (2008) 
        if air_temp >= 0:
            Psat_kPa = 0.61078*np.exp((17.27*air_temp)/(air_temp+237.3))
        else:
            Psat_kPa = 0.61078*np.exp((21.875*air_temp)/(air_temp+265.5))
        # Conversion kPa --> Pa
        Psat = 1000*Psat_kPa
        return Psat