#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 12:59:34 2023

@author: oluwaseunfadugba
"""
import numpy as np
import obspy
from obspy.core import UTCDateTime
import datetime
import pandas as pd
import statistics
import os
from glob import glob
import tsueqs_main_fns_Seun as tmf
import seaborn as sns
import matplotlib.pyplot as plt

# -------------------------------------------------------------------------------
def process_obs_data(obs_waveforms_dir,station_name,origin_time,fmax,stations):
    
    # Read file into a stream object
    disp_raw = obspy.read(obs_waveforms_dir + '/'+station_name+'.LXE.mseed')
    disp_raw += obspy.read(obs_waveforms_dir + '/'+station_name+'.LXN.mseed')
    
    disp_raw_copy = disp_raw.copy()
    
    # Correct a given time series stream object for the gain
    # Convert to m: (either cm/s/s to m/s/s, or cm to m):
    disp_raw_copy[0].data = disp_raw_copy[0].data/1.000000e+04/100 
    disp_raw_copy[1].data = disp_raw_copy[1].data/1.000000e+04/100
    
    # Get and remove the pre-event baseline
    # Define the baseline as the mean amplitude in the first 100 samples:
    disp_raw_copy[0].data = disp_raw_copy[0].data - np.mean(disp_raw_copy[0].data[0:100])
    disp_raw_copy[1].data = disp_raw_copy[1].data - np.mean(disp_raw_copy[1].data[0:100])
    
    # resample data and apply lowpass filter
    disp_raw_copy.resample(1)
    disp_raw_copy.filter('lowpass', freq=fmax, corners=2, zerophase=True)
    
    fill_value = disp_raw_copy[0].data[-60:].mean()
    disp_raw_copy.trim(starttime=UTCDateTime(origin_time),endtime=UTCDateTime(origin_time)+400, \
                              pad=True, nearest_sample=True,fill_value=fill_value)
    
    T_stream_obs = disp_raw_copy[0].copy()
    T_stream_obs.data = np.sqrt((disp_raw_copy[0].data)**2 + (disp_raw_copy[1].data)**2)
    T_stream_obs.stats.channel = 'LXT'
    
    # Save total disp mseed file
    outfile =  stations[:-1] + 'obs_t.sac.'+str(fmax)+'Hz'
    T_stream_obs.write(outfile, format='SAC')

    return T_stream_obs

# -------------------------------------------------------------------------------
def process_syn_data(integrate,is_talapas,origin_time,fmax,stations):
    
    # Read file into a stream object
    disp_raw = obspy.read(stations)
    disp_raw += obspy.read(stations[:-2]+'.n')
    
    disp_raw_copy = disp_raw.copy()
    
    #print(is_talapas)
    
    # For talapas, I have to remove 20 s from the start and end time. It's a mistake in the srf
    if is_talapas == 1:
        disp_raw_copy[0].stats.starttime = UTCDateTime(disp_raw_copy[0].stats.starttime)-20
        disp_raw_copy[1].stats.starttime = UTCDateTime(disp_raw_copy[1].stats.starttime)-20
        
    # resample data and apply lowpass filter
    disp_raw_copy.resample(1)
    disp_raw_copy.filter('lowpass', freq=fmax, corners=2, zerophase=True)
    
    if integrate == 1:
        disp_raw_copy.integrate();
        
    disp_raw_copy.trim(starttime=UTCDateTime(origin_time), endtime=UTCDateTime(origin_time)+400) 
        #,pad=True, nearest_sample=True,fill_value=np.NaN)
    
    T_stream_syn = disp_raw_copy[0].copy()
    T_stream_syn.data = np.sqrt((disp_raw_copy[0].data)**2 + (disp_raw_copy[1].data)**2)
    T_stream_syn.stats.channel = 'LXT'
    
    # Save total disp mseed file
    outfile =  stations[:-1] + 'syn_t.sac.'+str(fmax)+'Hz'
    T_stream_syn.write(outfile, format='SAC')

    return T_stream_syn

# -------------------------------------------------------------------------------
def calc_static_disp(stream,origin_time):
    
    # Extract noise
    noise_b4_shaking = stream.copy()
    noise_b4_shaking.trim(starttime=UTCDateTime(origin_time), endtime=UTCDateTime(origin_time)+10)

    if len(noise_b4_shaking.data)>=1:
        mean_noise_b4_shaking = statistics.mean(noise_b4_shaking.data)
    else:
        mean_noise_b4_shaking = 0.0

    # extract signal
    noise_after_shaking = stream.copy()
    e_time = stream.stats.endtime

    noise_after_shaking.trim(starttime=UTCDateTime(e_time)-60, endtime=UTCDateTime(e_time))

    # Check if noise after shaking exist
    if len(noise_after_shaking.data)>=1:
        mean_noise_after_shaking = statistics.mean(noise_after_shaking.data)
    else:
        mean_noise_after_shaking = 0.0
        
    static_disp = mean_noise_after_shaking - mean_noise_b4_shaking
    
    return static_disp

# ----------------------------------------------------------------------------------------------

def calc_SNR(stream,origin_time,hypdist):
    
    dp = datetime.timedelta(seconds=hypdist/6.5)
    
    # Extract noise
    noise_wf = stream.copy()
    noise_wf.trim(starttime=UTCDateTime(origin_time)-10+dp, endtime=UTCDateTime(origin_time)+dp)

    # extract signal
    signal_wf = stream.copy()
    signal_wf.trim(starttime=UTCDateTime(origin_time)+dp, endtime=UTCDateTime(origin_time)+120+dp)

    # determine SNR
    try:
        std_noise = statistics.pstdev(np.float64(noise_wf.data))
        std_signal = statistics.pstdev(np.float64(signal_wf.data))

        SNR = std_signal/std_noise 
    
    except:
        SNR = 0.0
            
    return SNR

# -------------------------------------------------------------------------------

def calc_xcorr(data_1,data_2):
    
    # data_1 = (data_1 - np.mean(data_1)) / (np.std(data_1))
    # data_2 = (data_2 - np.mean(data_2)) / (np.std(data_2))
    # cc = np.correlate(data_1,data_2, 'full')/ max(len(data_1), len(data_2))
    
    data_1 = data_1 / np.linalg.norm(data_1)
    data_2 = data_2 / np.linalg.norm(data_2)
    
    cc = np.correlate(data_1, data_2, mode = 'full')
    
    indexx = np.argmax(cc)
    xcorr = round(cc[indexx], 4)
    
    return xcorr
    
# --------------------------------------------------------------------------------------

def calc_pgd_tpgd(stream, origin_time):
    """
    From Tara's IM_fns_Tara.py
    Calculates pgd and time to peak intensity measure (IM) from origin time a
    """

    # Calculate pgd and time from origin 
    pgd = np.max(np.abs(stream.data))
    
    pgm_index = np.where(stream.data==pgd)[0][0]
    tpgd = stream.times(reftime=UTCDateTime(pd.to_datetime(origin_time)))[pgm_index]
    
    return(pgd, tpgd)

# ------------------------------------------------------------------------------------
    
def index_df_by_param(param,no_bins,opt_no_bins=0):

    if opt_no_bins == 1:
        param_bin_edges = np.histogram_bin_edges(param, bins=no_bins)
    else:
        param_bin_edges = np.arange(0,np.max(param)+no_bins,no_bins)
    
    param_bincenters = np.round(np.mean(np.vstack([param_bin_edges[0:-1],\
                                                   param_bin_edges[1:]]), axis=0),1)

    param_index = []
    
    #iterate over each element of the param
    for i in range(len(param)):
        
        # iterate over each bin
        for j in range(len(param_bin_edges)-1):#no_bins):
            if j == no_bins-1:
                if (param[i]>=param_bin_edges[j]) & (param[i]<=param_bin_edges[j+1]):
                    param_index = np.append(param_index,param_bincenters[j])
            else:
                if (param[i]>=param_bin_edges[j]) & (param[i]<param_bin_edges[j+1]):
                    param_index = np.append(param_index,param_bincenters[j])      
                     
    return param_index

# --------------------------------------------------------------------------------------

def plot_IM_res_vs_hypdist(flatfile_res_path,waveforms_dir,tag,fmax):

    # Plotting box plots
    sns.set_theme(style="ticks", color_codes=True)

    tick_font = 30
    zeroline_thick = 8
    #ylimm = 4.5
    label_font = 35
    
    boxprops_lw = 2
    capprops_lw = 2
    whiskerprops_lw = 2.5
    medianprops_lw = 3.5
    
    frame_lw = zeroline_thick/3
    grid_lw = zeroline_thick/5
    
    # ------------------------------------------------------------------------------------
    flatfile_res_dataframe = pd.read_csv(flatfile_res_path)   
     
        
    # ##################################################################################
    # Xcorr
    fig, axes = plt.subplots(2, 2, figsize=(25, 20))
    fig.suptitle('RESIDUAL PLOTS FOR '+tag.upper()+'('+str(fmax)+' Hz)',fontsize=label_font+10)

    #sns.boxplot(ax=axes[0, 0],x="hypdist_index", y="pgd_res",data=flatfile_res_dataframe,color='orange')
    sns.boxplot(ax=axes[0, 0],x="hypdist_index", y="pgd_res",data=flatfile_res_dataframe,color='orange',
                boxprops=dict(edgecolor='k',linewidth=boxprops_lw), #alpha=.6,
                capprops=dict(color='k', linewidth=2), #alpha=.6,
                whiskerprops=dict(color='k',ls = '--', linewidth= 2.5), #alpha=.7,
                medianprops={"linewidth":3.5},showfliers=True,linewidth=3.5) #"alpha":.6,"color":'k'
    
    # Increasing the linewidth of the frame border 
    for pos in ['right', 'top', 'bottom', 'left']:
        axes[0,0].spines[pos].set_linewidth(frame_lw)
    axes[0,0].grid(axis = 'y',color = 'k', linestyle = '--', linewidth = grid_lw)
    
    axes[0,0].axhline(y=0, ls='-',linewidth=zeroline_thick, color='r')
    axes[0,0].set_title("PGD Residuals", fontsize=label_font, fontdict={"weight": "bold"});
    axes[0,0].tick_params(axis='y',labelsize=tick_font,labelrotation=0)
    axes[0,0].set_ylabel('Ln Residual', fontsize=label_font)
    axes[0,0].axes.get_xaxis().set_visible(False)
    
    # ------------------------------------------------------------------------------------
    sns.boxplot(ax=axes[0, 1],x="hypdist_index", y="tPGD_res",data=flatfile_res_dataframe,color='orange',
                boxprops=dict(edgecolor='k',linewidth=boxprops_lw), #alpha=.6,
                capprops=dict(color='k', linewidth=capprops_lw), #alpha=.6,
                whiskerprops=dict(color='k',ls = '--', linewidth= whiskerprops_lw), #alpha=.7,
                medianprops={"linewidth":3.5},showfliers=True,linewidth=medianprops_lw) #"alpha":.6,"color":'k'
    
    # Increasing the linewidth of the frame border 
    for pos in ['right', 'top', 'bottom', 'left']:
        axes[0,1].spines[pos].set_linewidth(frame_lw)
    axes[0,1].grid(axis = 'y',color = 'k', linestyle = '--', linewidth = grid_lw)
    
    axes[0,1].axhline(y=0, ls='-',linewidth=zeroline_thick, color='r')
    axes[0,1].set_title("tPGD Residuals", fontsize=label_font, fontdict={"weight": "bold"});
    axes[0,1].tick_params(axis='y',labelsize=tick_font,labelrotation=0)
    axes[0,1].set_ylabel('Residual (s)', fontsize=label_font)
    axes[0,1].axes.get_xaxis().set_visible(False)
    
    # ------------------------------------------------------------------------------------
    sns.boxplot(ax=axes[1, 0],x="hypdist_index", y="sd_res",data=flatfile_res_dataframe,color='orange',
                boxprops=dict(edgecolor='k',linewidth=boxprops_lw), #alpha=.6,
                capprops=dict(color='k', linewidth=capprops_lw), #alpha=.6,
                whiskerprops=dict(color='k',ls = '--', linewidth= whiskerprops_lw), #alpha=.7,
                medianprops={"linewidth":3.5},showfliers=True,linewidth=medianprops_lw) #"alpha":.6,"color":'k'
    
    # Increasing the linewidth of the frame border 
    for pos in ['right', 'top', 'bottom', 'left']:
        axes[1,0].spines[pos].set_linewidth(frame_lw)
    axes[1,0].grid(axis = 'y',color = 'k', linestyle = '--', linewidth = grid_lw)
    
    axes[1,0].axhline(y=0, ls='-',linewidth=zeroline_thick, color='r')
    axes[1,0].set_title("Static Displ Residuals", fontsize=label_font, fontdict={"weight": "bold"});
    axes[1,0].tick_params(axis='x',labelsize=tick_font,labelrotation=45)
    axes[1,0].tick_params(axis='y',labelsize=tick_font,labelrotation=0)
    axes[1,0].set_xlabel('hypo_dist (km)', fontsize=label_font)
    axes[1,0].set_ylabel('Ln Residual', fontsize=label_font)

    # ------------------------------------------------------------------------------------
    sns.boxplot(ax=axes[1, 1],x="hypdist_index", y="xcorr",data=flatfile_res_dataframe,color='orange',
                boxprops=dict(edgecolor='k',linewidth=boxprops_lw), #alpha=.6,
                capprops=dict(color='k', linewidth=capprops_lw), #alpha=.6,
                whiskerprops=dict(color='k',ls = '--', linewidth= whiskerprops_lw), #alpha=.7,
                medianprops={"linewidth":3.5},showfliers=True,linewidth=medianprops_lw) #"alpha":.6,"color":'k'
    
    # Increasing the linewidth of the frame border 
    for pos in ['right', 'top', 'bottom', 'left']:
        axes[1,1].spines[pos].set_linewidth(frame_lw)
    axes[1,1].grid(axis = 'y',color = 'k', linestyle = '--', linewidth = grid_lw)
    
    axes[1,1].axhline(y=0, ls='-',linewidth=zeroline_thick, color='r')
    axes[1,1].set_title("Cross Correlation", fontsize=label_font, fontdict={"weight": "bold"});
    axes[1,1].tick_params(axis='x',labelsize=tick_font,labelrotation=45)
    axes[1,1].tick_params(axis='y',labelsize=tick_font,labelrotation=0)
    axes[1,1].set_xlabel('hypo_dist (km)', fontsize=label_font)
    axes[1,1].set_ylabel('xcorr', fontsize=label_font)

    # ------------------------------------------------------------------------------------
    figpath = os.getcwd() + '/'+waveforms_dir+'/fig.'+tag+'.IM_residuals_'+str(fmax)+'Hz.png'
    plt.savefig(figpath, bbox_inches='tight', dpi=300)
    
    plt.close()

    return

# --------------------------------------------------------------------------------------
def process_IM_residuals(eqparam,path,gflist,eqname,obs_waveforms_dir,waveforms_dir,flatfile_path,data):
    
    origin_time = eqparam[0]
    hyplon = eqparam[1]
    hyplat = eqparam[2]
    hypdepth = eqparam[3]
    mw = eqparam[4]
    integrate = eqparam[5]
    fmax = eqparam[6]
    
    # Read in the metadata file
    metadata = pd.read_csv(gflist, sep='\t', header=0,names=['st_name', 'lat', 'lon', 'elv', 
                                                         'samplerate', 'gain', 'units'])
    
    eventnames = np.array([]);       origintimes = np.array([])
    hyplons = np.array([]);          hyplats = np.array([])
    hypdepths = np.array([]);        mws = np.array([])
    stations_list = np.array([]);    stlons = np.array([])
    stlats = np.array([]);           stelevs = np.array([])
    hypdists = np.array([]);         rupt_no_list = np.array([])
    pgd_obs_list = np.array([]);     pgd_syn_list = np.array([])
    tpgd_syn_list = np.array([]);    tpgd_obs_list = np.array([])
    sd_syn_list = np.array([]);      sd_obs_list = np.array([])
    pgd_res_list = np.array([]);     tpgd_res_list = np.array([])
    sd_res_list = np.array([]);      xcorr_list = np.array([])
    SNR_obs_list = np.array([])
    
    # Looping over each ruptures
    for i in range(len(data[:,0])):
        
        print('Processing GNSS waveforms for ' + data[i,0])
        
        stations = np.array(sorted(glob(path + waveforms_dir + data[i,0]+'/*.sw4output/*.e'))) 
        
        # Looping over each station
        for j in range(len(stations)):  
            station_name = stations[j][-6:-2]        
            
            # Getting station info and calculate Pwave arrival time to the station
            sta_lat =  metadata.loc[(metadata['st_name'] == station_name), 'lat'].item() 
            sta_lon =  metadata.loc[(metadata['st_name'] == station_name), 'lon'].item()
            sta_elev =  metadata.loc[(metadata['st_name'] == station_name),'elv'].item()
    
            # Determining total horizontal waveforms
            T_stream_syn = process_syn_data(integrate,data[i,2],origin_time,fmax,stations[j])
            T_stream_obs = process_obs_data(obs_waveforms_dir,station_name,origin_time,fmax,stations[j])
    
    
    
    
    
    
    
    
    
    
    
    
            # Calculating waveform intensity measures
            # Determine hypodist 
            hypdist = tmf.compute_rhyp(sta_lon,sta_lat,sta_elev,hyplon,hyplat,hypdepth)
            
            # Determine SNR
            SNR_obs = calc_SNR(T_stream_obs,origin_time,hypdist)
            
            # Static displacement
            sd_obs = calc_static_disp(T_stream_obs,origin_time)
            sd_syn = calc_static_disp(T_stream_syn,origin_time)
    
            # Cross Correlation
            xcorr = calc_xcorr(T_stream_obs.data,T_stream_syn.data)
            
            # PGD and tPGD
            pgd_obs, tpgd_obs = calc_pgd_tpgd(T_stream_obs, origin_time)
            pgd_syn, tpgd_syn = calc_pgd_tpgd(T_stream_syn, origin_time)
            
            # Calculating the residuals
            pgd_res = np.log(pgd_obs/(pgd_syn+0.00001)) #0.001
            tpgd_res = tpgd_obs - tpgd_syn
            sd_res = np.log(np.abs(sd_obs/(sd_syn+0.00001)))
             
            # ###################################################################################
            # Append the earthquake, station info and intensity measure results for this station
            eventnames = np.append(eventnames, data[i,0])
            origintimes = np.append(origintimes, origin_time)
            
            hyplons = np.append(hyplons, hyplon)
            hyplats = np.append(hyplats, hyplat)
            hypdepths = np.append(hypdepths, hypdepth)
            mws = np.append(mws, mw)
            
            stations_list = np.append(stations_list, station_name) #str(str(sta_now).zfill(4)))
            stlons = np.append(stlons, sta_lon)
            stlats = np.append(stlats, sta_lat)
            stelevs = np.append(stelevs, sta_elev)
            
            hypdists = np.append(hypdists, round(hypdist,4))
            rupt_no_list = np.append(rupt_no_list, int(data[i,1]))
            
            pgd_obs_list = np.append(pgd_obs_list,round(pgd_obs,4))
            pgd_syn_list = np.append(pgd_syn_list,round(pgd_syn,4))
            pgd_res_list = np.append(pgd_res_list,pgd_res)
            
            tpgd_syn_list = np.append(tpgd_syn_list,tpgd_syn)
            tpgd_obs_list = np.append(tpgd_obs_list,tpgd_obs)
            tpgd_res_list = np.append(tpgd_res_list,tpgd_res)
            
            SNR_obs_list = np.append(SNR_obs_list,round(SNR_obs,4))
    
            sd_syn_list = np.append(sd_syn_list,sd_syn)
            sd_obs_list = np.append(sd_obs_list,sd_obs)
            sd_res_list = np.append(sd_res_list,sd_res)
            
            xcorr_list = np.append(xcorr_list,xcorr)
       
    
        # move corrected waveforms to corr_folder
        dest_folder_syn = stations[0][:-7]+'_t_syn.'+str(fmax)+'Hz'
        dest_folder_obs = stations[0][:-7]+'_t_obs.'+str(fmax)+'Hz'
        
        os.system('rm -rf ' + dest_folder_obs);
        os.system('rm -rf ' + dest_folder_syn);
        
        os.system('mkdir -p ' + dest_folder_obs);
        os.system('mkdir -p ' + dest_folder_syn);
        
        os.system('mv ' + stations[0][:-6] + '*obs_t.sac.'+str(fmax)+'Hz' + ' ' + dest_folder_obs);
        os.system('mv ' + stations[0][:-6] + '*syn_t.sac.'+str(fmax)+'Hz' + ' ' + dest_folder_syn);
        
    ######################### Put together dataframe ############################  
    # First, make a dictionary for main part of dataframe:
    dataset_dict = {'eventname':eventnames,     'rupt_no':rupt_no_list, 
                    'station':stations_list,    'origintime':origintimes,  
                    'hyplon':hyplons,           'hyplat':hyplats,
                    'hypdepth (km)':hypdepths,  'mw':mws,               
                    'stlon':stlons,             'stlat':stlats,            
                    'stelev':stelevs,           'hypdist':hypdists,
                    'pgd_obs':pgd_obs_list,     'pgd_syn':pgd_syn_list, 
                    'pgd_res':pgd_res_list,     'tPGD_obs':tpgd_obs_list,  
                    'tPGD_syn':tpgd_syn_list,   'tPGD_res':tpgd_res_list,
                    'xcorr':xcorr_list,         'SNR_obs':SNR_obs_list,
                    'sd_syn': sd_syn_list,      'sd_obs': sd_obs_list,
                    'sd_res': sd_res_list}    
        
    # Make main dataframe
    dataset_dict['station'] = dataset_dict['station'].astype(str)#.zfill(5)
    flatfile_res_df = pd.DataFrame(data=dataset_dict)
    
    # Index by hypdist and add it to the dataframe
    hypdist = np.array(flatfile_res_df['hypdist'])
    hypdist_index = index_df_by_param(hypdist,100)
    flatfile_res_df['hypdist_index'] = hypdist_index
    flatfile_res_df['hypdist_index'] = flatfile_res_df['hypdist_index'].astype(int)#.zfill(5)
    
    # Save df to file
    flatfile_res_df.to_csv(flatfile_path,index=False)    
        
    # plot residual plots       
    plot_IM_res_vs_hypdist(flatfile_path,waveforms_dir,eqname,fmax)
            
    return

# --------------------------------------------------------------------------------------
#%% def process_IM_residuals_1D(eqparam,path,gflist,eqname,obs_waveforms_dir,waveforms_dir,flatfile_path,data):
def process_IM_residuals_1D(eqparam,path,gflist,waveforms_dir):
    
    # Read in the metadata file
    metadata = pd.read_csv(gflist, sep='\t', header=0,names=['st_name', 'lat', 'lon', 'elv', 
                                                         'samplerate', 'gain', 'units'])
    
    for jj in range(len(eqparam)):

        # Extracting earthquake parameters
        eqname = eqparam[jj,0]
        origin_time = eqparam[jj,1]
        hyplon = eqparam[jj,2]
        hyplat = eqparam[jj,3]
        hypdepth = eqparam[jj,4]
        mw = eqparam[jj,5]
        integrate = eqparam[jj,6]
        fmax_all = eqparam[jj,7]
        obs_waveforms_dir = eqparam[jj,8]
        
        for ii in range(len(fmax_all)):
            
            fmax = fmax_all[ii]
            
            print('Processing GNSS waveforms for ' + eqname+' ('+str(fmax)+'Hz)')
            
            os.system('rm -rf ' + path + waveforms_dir+eqname+'/waveforms_'+str(fmax)+'Hz');
            os.system('mkdir -p ' + path + waveforms_dir+eqname+'/waveforms_'+str(fmax)+'Hz');
            
            flatfile_path = path +waveforms_dir+ 'flatfile_1d_'+eqname+'_'+str(fmax)+'Hz_Residuals.csv'
    
            # Initializa parameters
            eventnames = np.array([]);       origintimes = np.array([])
            hyplons = np.array([]);          hyplats = np.array([])
            hypdepths = np.array([]);        mws = np.array([])
            stations_list = np.array([]);    stlons = np.array([])
            stlats = np.array([]);           stelevs = np.array([])
            hypdists = np.array([]);         rupt_no_list = np.array([])
            pgd_obs_list = np.array([]);     pgd_syn_list = np.array([])
            tpgd_syn_list = np.array([]);    tpgd_obs_list = np.array([])
            sd_syn_list = np.array([]);      sd_obs_list = np.array([])
            pgd_res_list = np.array([]);     tpgd_res_list = np.array([])
            sd_res_list = np.array([]);      xcorr_list = np.array([])
            SNR_obs_list = np.array([])
            
            wfs_ruptures = np.array(sorted(glob(path + waveforms_dir+eqparam[jj,0]+'/waveforms/'+eqname+'*'))) 
            
            # Looping over each ruptures
            for i in range(len(wfs_ruptures)):
            
                #print(wfs_ruptures[i].split('/')[-1])
                
                # Renaming the sac files to .[e,n,z]
                stations_E = np.array(sorted(glob(wfs_ruptures[i]+'/*.LYE.sac'))) 
                stations_N = np.array(sorted(glob(wfs_ruptures[i]+'/*.LYN.sac'))) 
                stations_Z = np.array(sorted(glob(wfs_ruptures[i]+'/*.LYZ.sac'))) 
                
                if len(stations_E) != 0:
                    for jjj in range(len(stations_E)): 
                        os.system('mv '+stations_E[jjj] + ' '+stations_E[jjj][:-8]+'.e')
                        
                if len(stations_N) != 0:
                    for jjj in range(len(stations_N)):         
                        os.system('mv '+stations_N[jjj] + ' '+stations_N[jjj][:-8]+'.n')
                        
                if len(stations_Z) != 0:
                    for jjj in range(len(stations_Z)):         
                        os.system('mv '+stations_Z[jjj] + ' '+stations_Z[jjj][:-8]+'.z')
              
                
                # -----------------------------
                stations = np.array(sorted(glob(wfs_ruptures[i]+'/*.e'))) 
                
                # Looping over each station
                for j in range(len(stations)):  
                    station_name = stations[j][-6:-2]        
                    
                    # Getting station info and calculate Pwave arrival time to the station
                    sta_lat =  metadata.loc[(metadata['st_name'] == station_name), 'lat'].item() 
                    sta_lon =  metadata.loc[(metadata['st_name'] == station_name), 'lon'].item()
                    sta_elev =  metadata.loc[(metadata['st_name'] == station_name),'elv'].item()
            
                    #print(station_name)
                    # Determining total horizontal waveforms
                    T_stream_syn = process_syn_data(integrate,0,origin_time,fmax,stations[j])
                    T_stream_obs = process_obs_data(obs_waveforms_dir,station_name,origin_time,fmax,stations[j])
            
                    # Calculating waveform intensity measures
                    # Determine hypodist 
                    hypdist = tmf.compute_rhyp(sta_lon,sta_lat,sta_elev,hyplon,hyplat,hypdepth)
                    
                    # Determine SNR
                    SNR_obs = calc_SNR(T_stream_obs,origin_time,hypdist)
                    
                    # Static displacement
                    sd_obs = calc_static_disp(T_stream_obs,origin_time)
                    sd_syn = calc_static_disp(T_stream_syn,origin_time)
            
                    # Cross Correlation
                    xcorr = calc_xcorr(T_stream_obs.data,T_stream_syn.data)
                    
                    # PGD and tPGD
                    pgd_obs, tpgd_obs = calc_pgd_tpgd(T_stream_obs, origin_time)
                    pgd_syn, tpgd_syn = calc_pgd_tpgd(T_stream_syn, origin_time)
                    
                    # Calculating the residuals
                    pgd_res = np.log(pgd_obs/(pgd_syn+0.00001)) #0.001
                    tpgd_res = tpgd_obs - tpgd_syn
                    sd_res = np.log(np.abs(sd_obs/(sd_syn+0.00001)))
                     
                    # ###################################################################################
                    # Append the earthquake, station info and intensity measure results for this station
                    eventnames = np.append(eventnames, eqname)
                    origintimes = np.append(origintimes, origin_time)
                    
                    hyplons = np.append(hyplons, hyplon)
                    hyplats = np.append(hyplats, hyplat)
                    hypdepths = np.append(hypdepths, hypdepth)
                    mws = np.append(mws, mw)
                    
                    stations_list = np.append(stations_list, station_name) #str(str(sta_now).zfill(4)))
                    stlons = np.append(stlons, sta_lon)
                    stlats = np.append(stlats, sta_lat)
                    stelevs = np.append(stelevs, sta_elev)
                    
                    hypdists = np.append(hypdists, round(hypdist,4))
                    rupt_no_list = np.append(rupt_no_list, int(i))
                    
                    pgd_obs_list = np.append(pgd_obs_list,round(pgd_obs,4))
                    pgd_syn_list = np.append(pgd_syn_list,round(pgd_syn,4))
                    pgd_res_list = np.append(pgd_res_list,pgd_res)
                    
                    tpgd_syn_list = np.append(tpgd_syn_list,tpgd_syn)
                    tpgd_obs_list = np.append(tpgd_obs_list,tpgd_obs)
                    tpgd_res_list = np.append(tpgd_res_list,tpgd_res)
                    
                    SNR_obs_list = np.append(SNR_obs_list,round(SNR_obs,4))
            
                    sd_syn_list = np.append(sd_syn_list,sd_syn)
                    sd_obs_list = np.append(sd_obs_list,sd_obs)
                    sd_res_list = np.append(sd_res_list,sd_res)
                    
                    xcorr_list = np.append(xcorr_list,xcorr)
               
            
                # move corrected waveforms to corr_folder
                #print(stations)
                dest_folder_syn = stations[0][:-7]+'_t_syn.'+str(fmax)+'Hz'
                dest_folder_obs = stations[0][:-7]+'_t_obs.'+str(fmax)+'Hz'
                
                os.system('rm -rf ' + dest_folder_obs);
                os.system('rm -rf ' + dest_folder_syn);
                
                os.system('mkdir -p ' + dest_folder_obs);
                os.system('mkdir -p ' + dest_folder_syn);
                
                os.system('mv ' + stations[0][:-6] + '*obs_t.sac.'+str(fmax)+'Hz' + ' ' + dest_folder_obs);
                os.system('mv ' + stations[0][:-6] + '*syn_t.sac.'+str(fmax)+'Hz' + ' ' + dest_folder_syn);
                
                os.system('mv ' + dest_folder_obs + ' '+path + waveforms_dir+eqparam[jj,0]+
                          '/waveforms_'+str(fmax)+'Hz')
                os.system('mv ' + dest_folder_syn + ' '+path + waveforms_dir+eqparam[jj,0]+
                          '/waveforms_'+str(fmax)+'Hz')
            
            print(' ')
            ######################### Put together dataframe ############################  
            # First, make a dictionary for main part of dataframe:
            dataset_dict = {'eventname':eventnames,     'rupt_no':rupt_no_list, 
                            'station':stations_list,    'origintime':origintimes,  
                            'hyplon':hyplons,           'hyplat':hyplats,
                            'hypdepth (km)':hypdepths,  'mw':mws,               
                            'stlon':stlons,             'stlat':stlats,            
                            'stelev':stelevs,           'hypdist':hypdists,
                            'pgd_obs':pgd_obs_list,     'pgd_syn':pgd_syn_list, 
                            'pgd_res':pgd_res_list,     'tPGD_obs':tpgd_obs_list,  
                            'tPGD_syn':tpgd_syn_list,   'tPGD_res':tpgd_res_list,
                            'xcorr':xcorr_list,         'SNR_obs':SNR_obs_list,
                            'sd_syn': sd_syn_list,      'sd_obs': sd_obs_list,
                            'sd_res': sd_res_list}    
                
            # Make main dataframe
            dataset_dict['station'] = dataset_dict['station'].astype(str)#.zfill(5)
            flatfile_res_df = pd.DataFrame(data=dataset_dict)
            
            # Index by hypdist and add it to the dataframe
            hypdist = np.array(flatfile_res_df['hypdist'])
            hypdist_index = index_df_by_param(hypdist,100)
            flatfile_res_df['hypdist_index'] = hypdist_index
            flatfile_res_df['hypdist_index'] = flatfile_res_df['hypdist_index'].astype(int)#.zfill(5)
            
            # Save df to file
            flatfile_res_df.to_csv(flatfile_path,index=False)    
                
            # plot residual plots       
            plot_IM_res_vs_hypdist(flatfile_path,waveforms_dir,eqname,fmax)
            
    return