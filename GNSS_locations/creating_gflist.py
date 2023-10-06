#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 23 11:44:57 2022

@author: oluwaseunfadugba
"""

import obspy
from obspy.core import UTCDateTime
import statistics
import numpy as np
from glob import glob
import os
from os import environ,path,system
import matplotlib.pyplot as plt
import pandas as pd
import tsueqs_main_fns as tmf
import datetime        
import time
from shapely.geometry import Point, Polygon

current_dir = '/Users/oluwaseunfadugba/Documents/Projects/TsE_ValerieDiego/'+\
    'TsE_1D_vs_3D/1D_vs_3D_HR-GNSS_CrustalDeformation/GNSS_locations/'

data_dir = '/Users/oluwaseunfadugba/Documents/Projects/TsE_ValerieDiego/Data_GNSSdata'

snr_thresh = 3
data =np.array([['Ibaraki2011',    "2011-03-11T06:15:34", 141.2653, 36.1083, 43.2], 
                ['Iwate2011',      "2011-03-11T06:08:53", 142.7815, 39.8390, 31.7],
                ['Miyagi2011A',    "2011-03-09T02:45:12", 143.2798, 38.3285,  8.3],
                ['Tokachi2003',    "2003-09-25T19:50:06", 143.9040, 41.7750, 27.0]],dtype=object)

os.chdir(current_dir)
output_dir = current_dir

create_all_glist = 1
create_each_gflist = 1
plot_all_GNSS_loc = 1
plot_hist_hypodist = 1
plot_used_GNSS_with_rupts = 1

#%% Helper functions
def write_gflist_file(output_dir,eqnames,st_name, st_lat, st_lon):
    
    '''
    This function writes the GNSS station info to file.

    '''

    file = open(output_dir + '/'+ eqnames + '.gflist', "w")
    
    #file.write('GNSS Locations for '+ outfilename + ' Earthquake' + "\n")
    file.write('#station lon  lat "static,disp,vel,tsun,strain"'+"\n")
    
    for index in range(len(st_lat)):
    
        file.write(st_name[index] + "\t" + str(round(st_lon[index],4)) + "\t" + 
                   str(round(st_lat[index],4)) + "\t" + '0 1 0 0 0 0'+"\n")
        
    file.close()
    
    return

def write_ALL_GNSS_sta_to_file(output_dir,st_name, st_lat, st_lon, st_elv, 
                               st_samp_rate, st_gain, st_unit):

    '''
    This function writes the GNSS station info to file.

    '''

    file = open(output_dir + '/ALL_events.GNSS_locs.txt', "w")
    
    file.write('GNSS Locations for ALL the Earthquakes' + "\n")
    file.write('st_name' + "\t" + 'lat' + "\t" + 'lon' + "\t" + \
                   'elv' + "\t" + 'samp_rate' + "\t" + 'gain'+ "\t" + 'unit'+"\n")
    
    for index in range(len(st_lat)):
        file.write(st_name[index] + "\t" + st_lat[index] + "\t" + st_lon[index] + \
                   "\t" + st_elv[index] + "\t" + st_samp_rate[index] + "\t" + \
                   st_gain[index]+ "\t" + st_unit[index]+"\n")

    file.close()
    return

def plot_hist_hypodist(ax,eqname,hypolon,hypolat,gflist_filename):
    
    #ax = fig.add_subplot(a,b,c)
    
    # Read in the metadata file
    stadata = pd.read_csv(gflist_filename, sep='\t', header=0,names=['st_name', 'lon', 'lat', 'others'])

    # Initialize array
    dist_data = [];

    for i in range(len(stadata.index)):   
        stlat = stadata['lat'][i]
        stlon = stadata['lon'][i]

        #determine distance
        dist = tmf.compute_repi(stlon,stlat,hypolon,hypolat)

        dist_data.append(dist)

    print(max(dist_data))

    ax.hist(dist_data, bins = [0, 100, 200, 300,400, 500,
                        600, 700,800, 900, 1000,1100,1200, 1300, 1400,
                        1500, 1600, 1700,1800,1900,2000])

    label_font = 20
    ax.set_title(eqname, fontsize=label_font+3)
    ax.set_xlabel('hypo_dist (km)', fontsize=label_font)
    ax.set_ylabel('freq', fontsize=label_font)
    ax.tick_params(axis='x',labelsize=label_font,labelrotation=0)
    ax.tick_params(axis='y',labelsize=label_font,labelrotation=0)
    
    ax.grid()
    return

def extract_gflist(gflist_filename,poly_E,poly_W):
    
    # Read in the metadata file
    stadata = pd.read_csv(gflist_filename, sep='\t', header=0,names=['st_name', 'lon', 'lat', 'others'])

    # Initialize array
    lon = []; lat = []; 
    _lon = []; _lat = []; 
    
    #print(stadata)
    for i in range(len(stadata.index)-1):
        
        lat_pt = float(stadata['lat'][i+1])
        lon_pt = float(stadata['lon'][i+1])
                
        # Check if the station is within the polygons using the within function
        p1 = Point(lon_pt, lat_pt)
        
        if p1.within(poly_E) == True or p1.within(poly_W) == True:
            lon.append(lon_pt)
            lat.append(lat_pt)
    
        else:        
            _lon.append(lon_pt)
            _lat.append(lat_pt)
            
    return lon,lat,_lon,_lat


def extract_allgflist(gflist_filename,poly_E,poly_W):
    
    # Read in the metadata file
    stadata = pd.read_csv(gflist_filename, sep='\t', header=0,names=['st_name', 'lat', 'lon', 'elev','sr','gain','unit'])

    # Initialize array
    lon = []; lat = []; 
    _lon = []; _lat = []; 
    
    #print(stadata)
    for i in range(len(stadata.index)-1):
        
        lat_pt = float(stadata['lat'][i+1])
        lon_pt = float(stadata['lon'][i+1])
                
        # Check if the station is within the polygons using the within function
        p1 = Point(lon_pt, lat_pt)
        
        if p1.within(poly_E) == True or p1.within(poly_W) == True:
            lon.append(lon_pt)
            lat.append(lat_pt)
    
        else:        
            _lon.append(lon_pt)
            _lat.append(lat_pt)
            
    return lon,lat,_lon,_lat



def extract_source_pts(rupt):
    #Read mudpy file
    f=np.genfromtxt(rupt)
    
    lon_s = np.array([])
    lat_s = np.array([])
    depth_s = np.array([])
    
    #loop over subfaults
    for kfault in range(len(f)):

        zero_slip=False

        #Get subfault parameters
        lon=f[kfault,1]
        lat=f[kfault,2]
        depth=f[kfault,3]*1000 #in m for sw4
        strike=f[kfault,4]
        dip=f[kfault,5]
        area=f[kfault,10]*f[kfault,11] #in meters, cause this isn't dumb SRF
        #tinit=f[kfault,12]+time_pad
        #rake=rad2deg(arctan2(f[kfault,9],f[kfault,8]))
        slip=np.sqrt(f[kfault,8]**2+f[kfault,9]**2)
        rise_time=f[kfault,7]
        rigidity=f[kfault,13]

        #If subfault has zero rise time or zero slip
        zero_slip=False
        if slip==0:
            zero_slip=True
            #print('Zero slip at '+str(kfault))
        elif rise_time==0:
            slip=0
            zero_slip=True
            #print('Zero rise time at '+str(kfault))     

        #make rake be -180 to 180
        #if rake>180:
        #    rake=rake-360

        if zero_slip==False:
            
           lon_s = np.append(lon_s, lon)
           lat_s = np.append(lat_s, lat)
           depth_s = np.append(depth_s, depth)
           
    return lon_s,lat_s,depth_s


#%% Create txt file with all the GNSS stations from the .chan files
if create_all_glist == 1:    
    # Initialize variables
    st_names = []; st_lats = []; st_lons = []; st_elvs = [] 
    st_samp_rate = []; st_gain = []; st_unit = [] 
    n = 0
    
    # Extract GNSS locations for each earthquake and concatenate them
    for eqnames in data[:,0]:
        # concatenate infile name and open with numpy
        infile = data_dir + '/' + eqnames + '/' + eqnames + '_disp.chan'
        chan = np.genfromtxt(infile,skip_header=1,dtype='unicode')
        chan_z = chan[chan[:,3]=='LXZ',:]
        
        # extract parameters
        st_names = np.append(st_names,chan_z[:,1],axis = 0)
        st_lats = np.append(st_lats,chan_z[:,4],axis = 0)  
        st_lons = np.append(st_lons,chan_z[:,5],axis = 0)   
        st_elvs = np.append(st_elvs,chan_z[:,6],axis = 0)
        st_samp_rate = np.append(st_samp_rate,chan_z[:,7],axis = 0)
        st_gain = np.append(st_gain,chan_z[:,8],axis = 0)
        st_unit = np.append(st_unit,chan_z[:,9],axis = 0)
    
    # Extract parameters for only the unique stations
    st_name_all, st_names_index = np.unique(np.array(st_names), return_index=True)
    st_lat_all = [st_lats[i] for i in st_names_index]
    st_lon_all = [st_lons[i] for i in st_names_index]
    st_elv_all = [st_elvs[i] for i in st_names_index]  
    st_samp_rate_all = [st_samp_rate[i] for i in st_names_index]
    st_gain_all = [st_gain[i] for i in st_names_index]
    st_unit_all = [st_unit[i] for i in st_names_index]
    
    # write GNSS locations to file
    write_ALL_GNSS_sta_to_file(output_dir,st_name_all, st_lat_all, st_lon_all, \
                           st_elv_all, st_samp_rate_all, st_gain_all, st_unit_all)
        
#%% Create gflist for each earthuake using SNR_THRESH

if create_each_gflist == 1:
    
    # Determining SNR of the GNSS waveforms to remove stations below some threshold
    # Loop over each earthquake
    for j in range(len(data)):
        eqt_name = data[j,0] 
        origin_time = data[j,1] 
        
        hyplon = data[j,2] 
        hyplat = data[j,3] 
        hypdepth = data[j,4] 
    
        disp_files_T = np.array(sorted(glob(data_dir + '/' + eqt_name + 
                                            '/disp_corr/*.LXT.sac.corr')))
    
        # concatenate infile name
        infile = data_dir + '/ALL_events.GNSS_locs.txt'
        
        # Initialize variables
        st_names = []
        st_lats = []
        st_lons = []
    
        # Read in the metadata file
        metadata = pd.read_csv(infile, sep='\t', header=0,names=['st_name', 'lat', 'lon', 'elv', 
                                                             'samplerate', 'gain', 'units'])
    
        metadata.dropna(inplace = True)
        
        # loop over each total horizontal waveform
        for i in range(len(disp_files_T)): 
            sta_name = disp_files_T[i].split('/')[-1].split('.')[0]
    
            sta_lat =  metadata.loc[(metadata['st_name'] == int(sta_name)), 'lat']
            sta_lon =  metadata.loc[(metadata['st_name'] == int(sta_name)), 'lon']
            sta_elev =  metadata.loc[(metadata['st_name'] == int(sta_name)), 'elv']
        
            if sta_lat.empty==False & sta_lon.empty==False & sta_elev.empty==False:
                sta_lat = sta_lat.item()
                sta_lon = sta_lon.item()
                sta_elev = sta_elev.item()
    
                if bool(sta_lat)==True & bool(sta_lon)==True:
                    # Determine hypodist
                    hypdist = tmf.compute_rhyp(sta_lon,sta_lat,sta_elev,hyplon,hyplat,hypdepth)#6.5 
    
                    # Determine p_time
                    p_time = hypdist/6.5
                    dp = datetime.timedelta(seconds=p_time)
    
                    # Load raw waveforms for a particular GNSS station
                    data_raw = obspy.read(disp_files_T[i])
    
                    # Extract noise
                    noise_wf = data_raw.copy()
                    noise_wf.trim(starttime=UTCDateTime(origin_time)+dp-10, 
                                  endtime=UTCDateTime(origin_time)+dp)
    
                    # extract signal
                    signal_wf = data_raw.copy()
                    signal_wf.trim(starttime=UTCDateTime(origin_time)+dp, 
                                   endtime=UTCDateTime(origin_time)+120+dp)
    
                    # Check if noise and signal waveforms exist
                    if len(signal_wf)==1 & len(noise_wf)==1:
                        # determine SNR
                        #std_noise = statistics.pstdev(noise_wf[0].data)
                        #std_signal = statistics.pstdev(signal_wf[0].data)
                        std_noise = statistics.pstdev(np.float64(noise_wf[0].data))
                        std_signal = statistics.pstdev(np.float64(signal_wf[0].data))
                        
    
                        if std_noise != 0:
                            SNR = std_signal/std_noise 
    
                            if SNR >= snr_thresh:
                                st_names.append(sta_name)
                                st_lats.append(sta_lat)
                                st_lons.append(sta_lon)
    
        print(eqt_name)
        print(len(st_lats))
        print(st_names)
        write_gflist_file(output_dir,eqt_name.lower(),st_names, st_lats, st_lons)
        print('')
        print('')
        
    #------------------------------------------------------------------------------
    # create output directory
    os.system('mkdir -p gflists');
    os.system('mv *gflist *_locs.txt gflists');

#%% Plotting all GNSS stations including rfile domain
if plot_all_GNSS_loc == 1:
    gflist_filename = 'gflists/ALL_events.GNSS_locs.txt'          
    
    grid_E = [34,47,134.1125,147] # [lat1,lat2,lon1,lon2]
    grid_W = [30,37.9,129,141.1] # [lat1,lat2,lon1,lon2]
    
    # -----------------------------------------------------------
    # Read in the metadata file
    stadata = pd.read_csv(gflist_filename, sep='\t', header=0,\
                          names=['st_name', 'lat', 'lon', 'elev','sr','gain','unit'])
    
    # Create a Polygon with the coordinates of the East and West 3D velocity model domains
    coords_E = [(grid_E[2], grid_E[0]), (grid_E[3], grid_E[0]), 
                (grid_E[3], grid_E[1]), (grid_E[2], grid_E[1])]
    coords_W = [(grid_W[2], grid_W[0]), (grid_W[3], grid_W[0]), 
                (grid_W[3], grid_W[1]), (grid_W[2], grid_W[1])]
    
    poly_E = Polygon(coords_E)
    poly_W = Polygon(coords_W)
    
    # Initialize array
    lon = []; lat = []; elev = []
    _lon = []; _lat = []; _elev = []
    
    #print(stadata)
    for i in range(len(stadata.index)-1):
        
        lat_pt = float(stadata['lat'][i+1])
        lon_pt = float(stadata['lon'][i+1])
        elev_pt = float(stadata['elev'][i+1])
        
        # Check if the station is within the polygons using the within function
        p1 = Point(lon_pt, lat_pt)
        
        if p1.within(poly_E) == True or p1.within(poly_W) == True:
            lon.append(lon_pt)
            lat.append(lat_pt)
            elev.append(elev_pt)
    
        else:        
            _lon.append(lon_pt)
            _lat.append(lat_pt)
            _elev.append(elev_pt)
            
    print(len(stadata['lon']))
    print(len(lon))
    print(len(_lon))
    
    
    fig = plt.figure(figsize=(25, 15), dpi=100)
    ax1 = fig.add_subplot(1, 2,1)
    msize = 15
    
    plt.plot(lon, lat,'>',c='b',markersize = msize/2.5, \
             label='GNSS Stations (Used, '+str(len(lon))+')')
    plt.plot(_lon, _lat,'>',c='g',markersize = msize/2.5, \
             label='Stations (Not used, '+str(len(_lon))+')')
    
    # Plotting the west and east Japan 3D velocity boundaries
    plt.plot([129, 147, 147, 129, 129],[30, 30, 47, 47, 30], '--', \
              label='All 3DVel Domain', linewidth=5)
    plt.plot([134.1125, 147, 147, 134.1125, 134.1125],[34, 34, 47, 47, 34], '-', \
              label='East 3DVel Domain', linewidth=3)
    plt.plot([129, 141.1, 141.1, 129, 129],[30, 30, 37.9, 37.9, 30], '-', \
              label='West 3DVel Domain', linewidth=3)
    
    # Plotting the earthquake locations
    plt.plot(140.6727,36.9457, 'r-', marker=(5, 1), markersize=msize, \
              linestyle='None', label='Earthquakes')
    plt.plot(141.2653,36.1083, 'r-', marker=(5, 1), markersize=msize)
    plt.plot(142.7815,39.8390, 'r-', marker=(5, 1), markersize=msize)
    plt.plot(130.7630,32.7545, 'r-', marker=(5, 1), markersize=msize)
    plt.plot(143.2798,38.3285, 'r-', marker=(5, 1), markersize=msize)
    plt.plot(141.9237,38.2028, 'r-', marker=(5, 1), markersize=msize)
    plt.plot(144.8940,37.8367, 'r-', marker=(5, 1), markersize=msize)
    plt.plot(144.3153,37.8158, 'r-', marker=(5, 1), markersize=msize)
    plt.plot(144.5687,37.1963, 'r-', marker=(5, 1), markersize=msize)
    plt.plot(142.3720,38.2970, 'r-', marker=(5, 1), markersize=msize)
    plt.plot(143.9040,41.7750, 'r-', marker=(5, 1), markersize=msize)
    
    ax1.set_xlabel('lon (deg)',fontsize=msize+5)
    ax1.set_ylabel('lat (deg)',fontsize=msize+5)
    ax1.set_title('Station and Receiver Locations',fontsize=25)
    
    ax1.xaxis.set_tick_params(labelsize=msize+5,labelrotation=25)
    ax1.yaxis.set_tick_params(labelsize=msize+5)
    
    plt.legend(loc='lower right',fontsize=16)
    plt.grid()
    #plt.show()   
    
    # Saving figure to file
    figpath = os.getcwd() +'/fig.Domain_3D_Japan_velmodel.png'
    plt.savefig(figpath, bbox_inches='tight', dpi=200)


#%% Plot histogram of epicentral distance

if plot_hist_hypodist == 1:
    # Creating plot
    fig,axes = plt.subplots(2,3,figsize =(25, 18))
    fig.suptitle('Histogram of Epicentral Distance',fontsize=30)
    
    plot_hist_hypodist(axes[0,0],"Ibaraki 2011",141.2653,36.1083,'gflists/ibaraki2011.gflist')    
    plot_hist_hypodist(axes[0,1],"Iwate 2011",142.7815,39.8390,'gflists/iwate2011.gflist')    
    plot_hist_hypodist(axes[0,2],"Miyagi 2011A",143.2798,38.3285,'gflists/miyagi2011a.gflist')    
    plot_hist_hypodist(axes[1,2],"Tokachi 2003",143.9040,41.7750,'gflists/tokachi2003.gflist')    
    
    # ------------------------------------------------------------------------------------
    figpath = os.getcwd() +'/fig.hist_epicentral_dist.png'
    plt.savefig(figpath, bbox_inches='tight', dpi=300)
    
    plt.show()
    plt.close()

