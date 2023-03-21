#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  3 10:43:18 2022

@author: oluwaseunfadugba
"""

import pygmt, os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from string import ascii_lowercase as alphab
import time
start = time.time()

current_dir = '/Users/oluwaseunfadugba/Documents/Projects/TsE_ValerieDiego/'+\
    'TsE_1D_vs_3D/1D_vs_3D_HR-GNSS_CrustalDeformation/residual_map_3D/'
flatfile_res = current_dir+'flatfile_ibaraki2011_srcmod_srf3d_rupt5_fullrfile_talapas_0.25Hz_Residuals.csv'

os.chdir(current_dir)

tcb10 = ['#006BA4', '#FF800E', '#ABABAB', '#595959', '#5F9ED1', '#C85200', 
         '#898989', '#A2C8EC', '#FFBC79', '#CFCFCF']


#%% functions
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

def plot_histogram(data,comp,title):
    fig, axs = plt.subplots(1, 1, tight_layout=True,figsize=(15, 10))
    n_bins = 15
    plt.title(title, fontsize=70)
    
    tick_font =50
    # N is the count in each bin, bins is the lower-limit of the bin
    N, bins, patches = axs.hist(data, bins=n_bins)
    fracs = bins / bins.max() # We'll color code by height, but you could use any scalar
    norm = colors.Normalize(fracs.min(), fracs.max())# we need to normalize the data to 0..1 for the full range of the colormap
    for thisfrac, thispatch in zip(fracs, patches): # Now, we'll loop through our objects and set the color of each accordingly
        color = plt.cm.jet(norm(thisfrac))
        thispatch.set_facecolor(color)
    plt.xlabel('xcorr', fontsize=tick_font)
    plt.ylabel('freq', fontsize=tick_font)
    
    axs.tick_params(axis='x',labelsize=tick_font,labelrotation=0,length=15, width=3)
    axs.tick_params(axis='y',labelsize=tick_font,labelrotation=0,length=15, width=3)

    # Increasing the linewidth of the frame border 
    for pos in ['right', 'top', 'bottom', 'left']:
        plt.gca().spines[pos].set_linewidth(3)
    
    fig.savefig(current_dir+"/fig."+comp+"_xcorrhistogram_slant_ibaraki2.png")
    
    
    
def plot_map(comp,title):
    fig = pygmt.Figure()
    
    # Set the region for the plot to be slightly larger than the data bounds.
    region = [130,145,30,45]
     
    with pygmt.clib.Session() as session:
        session.call_module('gmtset', 'FONT 25p') #15.3
        session.call_module('gmtset', 'MAP_FRAME_TYPE fancy')
        session.call_module('gmtset', 'MAP_FRAME_WIDTH 0.25')
        
    grid = pygmt.datasets.load_earth_relief(resolution="10m", region=region)
     
    fig.coast(region=region,projection="M15c",land="gray",water="lightblue",borders="1/0.5p",
        shorelines="1/0.5p,black",frame="ag")
    
    #fig.grdimage(grid=grid, projection="M15c", frame="ag",cmap="geo")
    fig.basemap(frame=["a", '+t"'+title+'"'])
    

    def cs(x1,x2,x3,x4,x5):
        num = 7
        return np.concatenate((np.linspace(x1,x2,num), 
                              np.linspace(x2,x3,num),
                              np.linspace(x3,x4,num),
                              np.linspace(x4,x5,num)))
    
    homerupt = '/Users/oluwaseunfadugba/Documents/Projects/TsE_ValerieDiego/TsE_1D_vs_3D/1D_Modeling_using_FQs_Mudpy/Running_FakeQuakes_now/ibaraki2011_srcmod/ruptures/ibaraki2011_srcmod.000005.rupt'
    [lon_s,lat_s,depth_s] = extract_source_pts(homerupt); 
    fig.plot(x=lon_s,y=lat_s,style="c0.1",color="magenta")
    
    # Plotting the earthquake locations
    fig.plot(x=141.2653,y=36.1083,style="a0.7", color="red",pen="1p,black") ; 
    fig.text(x=141.2653,y=36.1083, text='                            Ibaraki 2011', font="16p,Helvetica-Bold,white")

    stadata_res = pd.read_csv(flatfile_res) 

    if comp == 'xcorr':
        pygmt.makecpt(cmap="jet", series=[stadata_res[comp].min(),1.0])#stadata_res[comp].max()
    else:
        rang = round(np.max([round(stadata_res[comp].min(),3), round(stadata_res[comp].max(),3)]),2)
        #print(rang)
        pygmt.makecpt(cmap="polar", series=[-rang,rang])
    
    
    #print(stadata_res['pgd_res'])
    fig.plot(x=stadata_res['stlon'],y=stadata_res['stlat'],style="t0.3",color=stadata_res[comp],
             pen="0.9p,black",transparency=0,cmap=True,label='GNSS_Stations')
    
    if comp == 'pgd_res':
        s = 'PGD Residual (ln)';
    elif comp == 'tPGD_res':
        s = 'tPGD Residual (s)';
    elif comp == 'sd_res':
        s = 'Static Displacement Residual (ln)';
    elif comp == 'xcorr':
        s = 'xcorr (value)';
        
    fig.colorbar(frame='af+l"'+s+'"')
    
    #fig.legend()
    fig.show()
    fig.savefig(current_dir+"/fig."+comp+"_ibaraki_srcmod.png",dpi=1000)


#%% Driver

plot_map('pgd_res','PGD Residual (SW4 3D)')
plot_map('tPGD_res','tPGD Residual (SW4 3D)')
plot_map('sd_res','Static Displacement Residual (SW4 3D)')
plot_map('xcorr','Cross Correlation (SW4 3D)')


# ####################################################################
end = time.time()
time_elaps = end - start
if time_elaps < 60:
    print(f'Duration: {round(time_elaps)} seconds')
else:
    print(f'Duration: {round(time_elaps/60)} minutes')
