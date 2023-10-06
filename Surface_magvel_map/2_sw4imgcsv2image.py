#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  2 14:31:24 2023

@author: oluwaseunfadugba

To use this code, change the environment to pygmt by running 
"conda activate pygmt" on a new terminal and restart Jupyter notebook.
# conda activate /Users/oluwaseunfadugba/mambaforge/envs/pygmt
"""

import numpy as np
import pygmt
import pandas as pd
import os
import time
from matplotlib import pyplot as plt
import matplotlib
from matplotlib.colors import LinearSegmentedColormap
from glob import glob
import imageio
import moviepy.editor as mp
import cv2

cwd = '/Users/oluwaseunfadugba/Documents/Projects/TsE_ValerieDiego/TsE_1D_vs_3D/'+\
    '1D_vs_3D_HR-GNSS_CrustalDeformation/Surface_magvel_map/'
os.chdir(cwd)
start = time.time()

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

def plot_map(region,outputfilename):

    evlon=141.2653; evlat=36.1083
    rupt_5 = '/Users/oluwaseunfadugba/Documents/Projects/TsE_ValerieDiego/TsE_1D_vs_3D/'+\
        '1D_Modeling_using_FQs_Mudpy/Running_FakeQuakes_now/ibaraki2011_srcmod/ruptures/ibaraki2011_srcmod.000005.rupt'
    flatfile_res_path='/Users/oluwaseunfadugba/Documents/Projects/TsE_ValerieDiego/'+\
        'TsE_1D_vs_3D/1D_Modeling_using_FQs_Mudpy/IM_Residuals/Results_ibaraki2011_'+\
            'srcmod_IM_residuals/Flatfiles_IMs_ibaraki2011_srcmod.csv'    
    dist_thresh = 1000
    eqname = 'Ibaraki 2011'
    
    fig = pygmt.Figure()

    with pygmt.clib.Session() as session:
        session.call_module('gmtset', 'FONT 15.3p')
        session.call_module('gmtset', 'MAP_FRAME_TYPE fancy')
        session.call_module('gmtset', 'MAP_FRAME_WIDTH 0.25')
        
    grid = pygmt.datasets.load_earth_relief(resolution="15s", region=region)
    fig.grdimage(grid=grid, projection="M15c", frame=None,cmap="geo")#
  
    # plot ruptures
    [lon_s,lat_s,depth_s] = extract_source_pts(rupt_5); 
    fig.plot(x=lon_s,y=lat_s,style="c0.1",color="magenta")
    
    # Plotting the earthquake locations
    fig.plot(x=evlon,y=evlat,style="a0.7", color="red",pen="1p,black") ; 
    fig.text(x=evlon,y=evlat, text='                            '+eqname, font="16p,Helvetica-Bold,white")

    stadata = pd.read_csv(flatfile_res_path) 
    stadata = stadata[stadata['rupt_no']==1]
    stadata = stadata[stadata['hypdist']<=dist_thresh]
    
    fig.savefig(outputfilename,dpi=2000)

    return

def gif2mp4(gif):
    
    clip = mp.VideoFileClip(gif)
    clip.write_videofile(gif+'.mp4')

    return

# ####################################################################################

# Set the region for the plot to be slightly larger than the data bounds.
region = [130,145.5,32,45]
hotstart = 300

# Create basemap
outputfilename= 'fig.map_full_iba_forwaveprop.png'

if os.path.exists(outputfilename)==False:
    print('Creating basemap')
    plot_map(region,outputfilename)
else:
    print('Basemap exists')
    
# Overlay wave propagation on the basemap
if hotstart == 0:
    os.system('rm -rf surf_magvel_png')
    os.system('mkdir surf_magvel_png')

cdict = {'red':  ((0., 1, 1), (0.03, 1, 1), (0.20, 0, 0), (0.66, 1, 1), (0.89, 1, 1), (1, 0.5, 0.5)),
         'green':((0., 1, 1), (0.03, 1, 1), (0.20, 0, 0), (0.375, 1,1), (0.64, 1, 1), (0.91, 0, 0), (1, 0, 0)),
         'blue': ((0., 1, 1), (0.08, 1, 1), (0.20, 1, 1), (0.34, 1, 1), (0.65, 0, 0), (1, 0, 0))}
whitejet = matplotlib.colors.LinearSegmentedColormap('whitejet',cdict,256)

# Looping over surface magvel file
for tim in range(hotstart,365,5): #0,365,5
    t = str(tim)
    
    print('Processing time = '+t+' s')
    
    fig, ax = plt.subplots()
    plt.rcParams["figure.figsize"] = [7.00, 3.50]
    plt.rcParams["figure.autolayout"] = True
    im = plt.imread(outputfilename)

    im = ax.imshow(im, extent=region,alpha=1)

    # get colormap and create a colormap object
    ncolors = 256
    color_array = plt.get_cmap(whitejet)(range(ncolors))
    color_array[:,-1] = np.linspace(0.0,1.0,ncolors) # change alpha values
    map_object = LinearSegmentedColormap.from_list(name='whitejet_alpha',colors=color_array)
    plt.register_cmap(cmap=map_object) # register this new colormap with matplotlib

    stadata = pd.read_csv('surf_magvel_csv/surf_magvel_at_time_'+t+'s.csv')
    im=ax.imshow(stadata,extent=region, cmap='whitejet_alpha', vmin=0, vmax=0.08,alpha=6)
    
    cbar=fig.colorbar(im,fraction=0.03, pad=0.02) #0.02
    cbar.set_label('Surface Mag Velocity (m/s)') 
    
    fontsize = 14
    ax.set_aspect(1.5) 
    ax.set_xticks([130,135,140,145])
    ax.set_yticks([35,40,45])
    ax.set_ylabel('latitude',fontsize=fontsize)
    ax.set_xlabel('longitude',fontsize=fontsize)
    ax.tick_params(labelsize=fontsize)
    ax.set_title('time = '+t+' s',fontsize=fontsize+5)
    #ax.text(134, 44.3, 'time = '+t+' s', color='r', fontsize=fontsize, weight='bold')
    
    plt.close(fig)
    fig.savefig('surf_magvel_png/fig.surf_magvel_at_time_'+t.zfill(4)+'s.png',dpi=500,\
        bbox_inches='tight',facecolor='white', edgecolor='none')

# Creating gif from the figures
png_list = np.array(sorted(glob('surf_magvel_png/fig.surf_magvel_at_time_*s.png')))
fname_gif = 'surf_magvel.gif'
fname_mp4 = 'surf_magvel.mp4'

with imageio.get_writer(fname_gif, mode='I') as writer:
    for filename in png_list:
        image = imageio.imread(filename)
        writer.append_data(image)

# Converting png files to mp4
img_array = []
for filename in png_list:
    img = cv2.imread(filename)
    height, width, layers = img.shape
    size = (width,height)
    img_array.append(img)
 
out = cv2.VideoWriter(fname_mp4,cv2.VideoWriter_fourcc(*'DIVX'), 15, size)
 
for i in range(len(img_array)):
    out.write(img_array[i])
out.release()


# ####################################################################
end = time.time()
time_elaps = end - start
if time_elaps < 60:
    print(f'Duration: {round(time_elaps)} seconds')
else:
    print(f'Duration: {round(time_elaps/60)} minutes')