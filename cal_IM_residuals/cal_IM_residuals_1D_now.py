#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 12:22:01 2023

@author: oluwaseunfadugba
"""
# This function can only run under the mtspec environment
# To activate, close the Jupyter Notebook, run the command below and 
# restart the Jupyter Notebook
#     'conda activate mtspec'
# # For talapas, I have to remove 20 s from the start and end time. It's a mistake in the srf

import os

path = '/Users/oluwaseunfadugba/Documents/Projects/TsE_ValerieDiego/TsE_1D_vs_3D/'+\
    '1D_Modeling_using_FQs_Mudpy/'
os.chdir(path)

os.sys.path.insert(0, '/Users/oluwaseunfadugba/Documents/Projects/TsE_ValerieDiego/'+\
                   'TsE_1D_vs_3D/3D_Modeling_using_SW4/Running_Simulations/4_Model_results/')

import numpy as np
import time
import cal_IM_residuals_3D_now_funcs as funcs

start = time.time()

#%% 
wfs = 'Running_FakeQuakes_now/'
gflist = path + 'ALL_events.GNSS_locs.txt'
obs_wfs_ib = '/Users/oluwaseunfadugba/Documents/Projects/TsE_ValerieDiego/Data_GNSSdata/Ibaraki2011/disp'
obs_wfs_iw = '/Users/oluwaseunfadugba/Documents/Projects/TsE_ValerieDiego/Data_GNSSdata/Iwate2011/disp'
obs_wfs_mi = '/Users/oluwaseunfadugba/Documents/Projects/TsE_ValerieDiego/Data_GNSSdata/Miyagi2011A/disp'
obs_wfs_to = '/Users/oluwaseunfadugba/Documents/Projects/TsE_ValerieDiego/Data_GNSSdata/Tokachi2003/disp'

fmax= [0.25,0.49]; # folder_name, origin time, lon, lat, depth,  mag, integrate, fmax
eqparam =np.array([#['ibaraki2011_srcmod',"2011-03-11T06:15:34",141.2653,36.1083,43.2,7.9,0,fmax,obs_wfs_ib], 
                   #['ibaraki2011_zheng1',"2011-03-11T06:15:34",141.2653,36.1083,43.2,7.9,0,fmax,obs_wfs_ib],
                   ['iwate2011_zheng1',   "2011-03-11T06:08:53",142.7815,39.8390,31.7,7.4,0,fmax,obs_wfs_iw],
                   ['miyagi2011a_zheng1', "2011-03-09T02:45:12",143.2798,38.3285,8.3, 7.3,0,fmax,obs_wfs_mi],
                   ['miyagi2011a_usgs',   "2011-03-09T02:45:12",143.2798,38.3285,8.3, 7.3,0,fmax,obs_wfs_mi],
                   ['tokachi2003_srcmod1',"2003-09-25T19:50:06",143.9040,41.7750,27.0,8.3,0,fmax,obs_wfs_to],
                   ['tokachi2003_srcmod2',"2003-09-25T19:50:06",143.9040,41.7750,27.0,8.3,0,fmax,obs_wfs_to],
                   ['tokachi2003_srcmod3',"2003-09-25T19:50:06",143.9040,41.7750,27.0,8.3,0,fmax,obs_wfs_to],
                   ['tokachi2003_usgs',   "2003-09-25T19:50:06",143.9040,41.7750,27.0,8.3,0,fmax,obs_wfs_to]
                  ],dtype=object) 

funcs.process_IM_residuals_1D(eqparam,path,gflist,wfs)
print(' ')

#%% ####################################################################
end = time.time()
time_elaps = end - start
if time_elaps < 60:
    print(f'Duration: {round(time_elaps)} seconds')
else:
    print(f'Duration: {round(time_elaps/60)} minutes')

      
#ibaraki - 700/728
#iwate - good
#miyagi - 163/198
#tokachi - 121/236
