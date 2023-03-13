#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 17 10:38:35 2022

@author: oluwaseunfadugba
"""

#%%
def plot_obs_GNSS_SNR3_for_all_eqs(data_dir,eqt_name,origin_time,hyplon,hyplat,hypdepth):

    # For all Earthquake

    import obspy
    from obspy.core import UTCDateTime
    import statistics
    import numpy as np
    from glob import glob
    from os import environ
    import pandas as pd
    import tsueqs_main_fns_Seun as tmf
    import datetime
    import matplotlib.pyplot as plt
    import math
    import time

    # concatenate infile name
    infile = '/Users/oluwaseunfadugba/Documents/Projects/TsE_ValerieDiego/'+\
        'TsE_1D_vs_3D/1D_Modeling_using_FQs_Mudpy/Creating_Inputs/GNSS_locations/gflists/' + eqt_name.lower()+'.gflist'
    
    # Initialize variables
    st_names = []
    st_lats = []
    st_lons = []

    # Read in the metadata file
    metadata = pd.read_csv(infile, sep='\t', header=0,names=['st_name', 'lon', 'lat', 'others'])

    fig = plt.figure()
    fig.set_size_inches(185, 105)
    fig.suptitle('Observed GNSS waveforms for ' + eqt_name + ' at each station \n' \
                      + 'Subplot title represents Station name (corresponding SNR)', fontsize=160)
    
    # loop over each total horizontal waveform
    for i in range(len(metadata['st_name'])): 

        sta_name = str(metadata['st_name'][i]).zfill(4)

        sta_lat =  metadata.loc[(metadata['st_name'] == int(sta_name)), 'lat'].item()
        sta_lon =  metadata.loc[(metadata['st_name'] == int(sta_name)), 'lon'].item()
        sta_elev =  0 

        # Determine hypodist
        hypdist = tmf.compute_rhyp(sta_lon,sta_lat,sta_elev,hyplon,hyplat,hypdepth) 

        # Determine p_time
        p_time = hypdist/6.5
        dp = datetime.timedelta(seconds=p_time)

        # Load GNSS waveforms
        obs_disp_E = obspy.read(data_dir + '/' + eqt_name + '/disp_corr/'+ sta_name +'.LXE.sac.corr')
        obs_disp_N = obspy.read(data_dir + '/' + eqt_name + '/disp_corr/'+ sta_name +'.LXN.sac.corr')
        obs_disp_Z = obspy.read(data_dir + '/' + eqt_name + '/disp_corr/'+ sta_name +'.LXZ.sac.corr')
        obs_disp_T = obspy.read(data_dir + '/' + eqt_name + '/disp_corr/'+ sta_name +'.LXT.sac.corr')

        # Plot waveforms       
        linewd = 1.5
        NN = math.ceil(np.sqrt(len(metadata['st_name'])))
        ax = fig.add_subplot(NN, NN, i+1)

        # T
        ax.plot(obs_disp_T[0].times("matplotlib"), obs_disp_T[0].data, "r-",linewidth=linewd)
        ax.text(obs_disp_T[0].stats.starttime-10, obs_disp_T[0].data[0], 'T',color='r', fontsize=30)
        # Z
        sca_1 = 2.0*max(abs(obs_disp_T[0].data))
        ax.plot(obs_disp_Z[0].times("matplotlib"), obs_disp_Z[0].data+sca_1, "g-",linewidth=linewd)
        ax.text(obs_disp_Z[0].stats.starttime-10, obs_disp_Z[0].data[0]+sca_1, 'Z',color='g', fontsize=30)
        # N
        sca_2 = sca_1 + 2.0*max(abs(obs_disp_Z[0].data))
        ax.plot(obs_disp_N[0].times("matplotlib"), obs_disp_N[0].data+sca_2, "b-",linewidth=linewd)
        ax.text(obs_disp_N[0].stats.starttime-10, obs_disp_N[0].data[0]+sca_2, 'N',color='b', fontsize=30)
        # E
        sca_3 = sca_2 + 2.0*max(abs(obs_disp_N[0].data))
        ax.plot(obs_disp_E[0].times("matplotlib"), obs_disp_E[0].data+sca_3, "c-",linewidth=linewd)
        ax.text(obs_disp_E[0].stats.starttime-10, obs_disp_E[0].data[0]+sca_3, 'E',color='c', fontsize=30)

        ax.xaxis_date()
        fig.autofmt_xdate()
        ax.xaxis.set_tick_params(labelsize=30)
        ax.yaxis.set_tick_params(labelsize=30)

        #ax.axvline(x=UTCDateTime(origin_time), c ='k',ls = '--')
        ax.axvline(x=UTCDateTime(origin_time)+dp, c ='k',ls = '--')
        ax.axvline(x=UTCDateTime(origin_time)+dp+120, c ='k',ls = '--')
        ax.axvline(x=UTCDateTime(origin_time)+dp+180, c ='k',ls = '--')
        ax.axvline(x=UTCDateTime(origin_time)+dp+240, c ='k',ls = '--')
        #ax.axvline(x=UTCDateTime(origin_time)+400, c ='k',ls = '--')

        ax.text(UTCDateTime(origin_time), obs_disp_T[0].data[0], 'origin',color='k', fontsize=12)
        ax.text(UTCDateTime(origin_time)+dp, obs_disp_T[0].data[0], 'Parr',color='k', fontsize=12)
        ax.text(UTCDateTime(origin_time)+dp+120, obs_disp_T[0].data[0], 'P+120s',color='k', fontsize=12)
        ax.text(UTCDateTime(origin_time)+dp+180, obs_disp_T[0].data[0], 'P+180s',color='k', fontsize=12)
        ax.text(UTCDateTime(origin_time)+dp+240, obs_disp_T[0].data[0], 'P+240s',color='k', fontsize=12)
        #ax.text(UTCDateTime(origin_time)+400, obs_disp_T[0].data[0], 'T400s',color='k', fontsize=12)
        
        # Extract noise
        noise_wf = obs_disp_T.copy()
        noise_wf.trim(starttime=UTCDateTime(origin_time)-10+dp,endtime=UTCDateTime(origin_time)+dp)

        # extract signal
        signal_wf = obs_disp_T.copy()
        signal_wf.trim(starttime=UTCDateTime(origin_time)+dp,endtime=UTCDateTime(origin_time)+120+dp)

        # Check if noise and signal waveforms exist
        if len(signal_wf)==1 & len(noise_wf)==1:
            # determine SNR
            std_noise = statistics.pstdev(np.float64(noise_wf[0].data))
            std_signal = statistics.pstdev(np.float64(signal_wf[0].data))
            
            if std_noise != 0:
                SNR = std_signal/std_noise 
                if SNR >= 3:
                    st_names.append(sta_name)
                    st_lats.append(sta_lat)
                    st_lons.append(sta_lon)

            ax.set_title(sta_name + ' (' + str(round(SNR,1)) +')', fontsize=20)

    fig.savefig('fig.' + eqt_name + '_SNR3_observed_wfs.pdf', dpi=200)

    print(eqt_name)
    print(origin_time)
    #print(len(st_lats))
    #print(st_names)
    print('')
    print('')

    return

#%%

import obspy
from obspy.core import UTCDateTime
import obspy
# import statistics
# import numpy as np
# from glob import glob
import os 
# import pandas as pd
# import tsueqs_main_fns_Seun as tmf
# import datetime
import matplotlib.pyplot as plt
# import math
# import time

current_dir = '/Users/oluwaseunfadugba/Documents/Projects/TsE_ValerieDiego/'+\
    'TsE_1D_vs_3D/Conferences/TsE_Paper/Paper_Figures/Residuals_cartoon'

os.chdir(current_dir)

sta = '0042'
file_obs = '/Users/oluwaseunfadugba/Documents/Projects/TsE_ValerieDiego/Data_GNSSdata/Ibaraki2011/disp_corr/'+sta+'.LXT.sac.corr'
file_syn = '/Users/oluwaseunfadugba/Documents/Projects/TsE_ValerieDiego/TsE_1D_vs_3D/1D_Modeling_using_FQs_Mudpy/Running_FakeQuakes_now/ibaraki2011_srcmod/waveforms_corr/ibaraki2011_srcmod.000000/'+sta+'.LYT.sac.corr'
origin_time = "2011-03-11T06:15:34"

syn_disp_T = obspy.read(file_syn)
obs_disp_T = obspy.read(file_obs)

fig = plt.figure()
fig.set_size_inches(18, 10)
# fig.suptitle('Observed GNSS waveforms for ' + eqt_name + ' at each station \n' \
#                   + 'Subplot title represents Station name (corresponding SNR)', fontsize=160)

# Plot waveforms       
linewd = 5
fontsize = 40

ax = fig.add_subplot(1, 1, 1)

ax.plot(obs_disp_T[0].times("matplotlib")+15/24/3600, obs_disp_T[0].data, "k-",linewidth=linewd,label='observed')
ax.plot(syn_disp_T[0].times("matplotlib"), syn_disp_T[0].data, "r--",linewidth=linewd,label='synthetic')

ax.xaxis_date()
fig.autofmt_xdate()

ax.set_xlabel('time',fontsize=fontsize)
ax.set_ylabel('T-comp',fontsize=fontsize)
ax.set_ylim([0,0.5])
ax.grid()
ax.xaxis.set_tick_params(labelsize=fontsize)
ax.yaxis.set_tick_params(labelsize=fontsize)

ax.axvline(x=UTCDateTime(origin_time), c ='k',ls = '-.',linewidth=linewd,label='Origin time')
ax.legend(fontsize=fontsize-5)
fig.savefig('fig.residuals_cartoon.png', dpi=200)
