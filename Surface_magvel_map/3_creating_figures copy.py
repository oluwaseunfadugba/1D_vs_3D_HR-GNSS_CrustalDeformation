#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  2 19:59:38 2023

@author: oluwaseunfadugba
"""

#import libraries
import math
from matplotlib import pyplot as plt
import numpy as np
import matplotlib
from string import ascii_lowercase as alphab
from matplotlib.colors import LinearSegmentedColormap
import time
import pandas as pd
start = time.time()

fig_dir = '/Users/oluwaseunfadugba/Documents/Projects/TsE_ValerieDiego/TsE_1D_vs_3D/'+\
    '1D_vs_3D_HR-GNSS_CrustalDeformation/Surface_magvel_map/surf_magvel_png/'

#fig_nos = [0,10,30,50,70,100,120,150,170,200,220, 250,270,300,330,360]
t = [0,30,70,100,120,140,170,200,250,270,300,360] #330,



# basemap
outputfilename= 'fig.map_full_iba_forwaveprop.png'
region = [130,145.5,32,45]

cdict = {'red':  ((0., 1, 1), (0.03, 1, 1), (0.20, 0, 0), (0.66, 1, 1), (0.89, 1, 1), (1, 0.5, 0.5)),
         'green':((0., 1, 1), (0.03, 1, 1), (0.20, 0, 0), (0.375, 1,1), (0.64, 1, 1), (0.91, 0, 0), (1, 0, 0)),
         'blue': ((0., 1, 1), (0.08, 1, 1), (0.20, 1, 1), (0.34, 1, 1), (0.65, 0, 0), (1, 0, 0))}
whitejet = matplotlib.colors.LinearSegmentedColormap('whitejet',cdict,256)

# get colormap and create a colormap object
ncolors = 256
color_array = plt.get_cmap(whitejet)(range(ncolors))
color_array[:,-1] = np.linspace(0.0,1.0,ncolors) # change alpha values
map_object = LinearSegmentedColormap.from_list(name='whitejet_alpha',colors=color_array)
plt.register_cmap(cmap=map_object) # register this new colormap with matplotlib

# create figure
fig = plt.figure(figsize=(23, 20)) #,constrained_layout=True 20,17
fig.suptitle('Surface Magnitude Velocity for Ibaraki 2011 (SRCMOD Rupture 5)', fontsize=30)
  
for i in range(len(t)):
    # # reading images
    # Image1 = plt.imread(fig_dir+'fig.surf_magvel_at_time_'+str(fig_nos[i]).zfill(4)+'s.png')
    
    fig.add_subplot(3, 4, i+1) #,aspect=1.5
    # plt.imshow(Image1)
    # plt.axis('off')
    plt.text(125, 45.5, '('+alphab[i].upper()+')', color='k', fontsize=25)#, weight='bold')
    #plt.tight_layout()
      
    im = plt.imread(outputfilename)

    im = plt.imshow(im, extent=region,alpha=1)

    stadata = pd.read_csv('surf_magvel_csv/surf_magvel_at_time_'+str(t[i])+'s.csv')
    im=plt.imshow(stadata,extent=region, cmap='whitejet_alpha', vmin=0, vmax=0.08,alpha=6)
    
    fontsize = 20
    #plt.aspect(1.5) 
    #plt.gca().set_aspect(1.5)
    plt.xticks([130,135,140,145],fontsize=fontsize)
    plt.yticks([35,40,45],fontsize=fontsize)
    plt.ylabel('latitude',fontsize=fontsize)
    
    if i >= 8:
        plt.xlabel('longitude',fontsize=fontsize)
    plt.tick_params(labelsize=fontsize)
    plt.title('time = '+str(t[i])+' s',fontsize=fontsize+5)
    
    if i == 9:
        #cbar=fig.colorbar(im,fraction=0.03, pad=0.5) #0.02
        cbar = fig.colorbar(im, orientation="horizontal", pad = 0.5, shrink=1.6)
        cbar.set_label('Surface Mag Velocity') 

#plt.close(fig)
plt.show()
fig.savefig('fig8_surf_magvel_snapshots.png',dpi=500,\
    bbox_inches='tight',facecolor='white', edgecolor='none')


# ####################################################################
end = time.time()
time_elaps = end - start
if time_elaps < 60:
    print(f'Duration: {round(time_elaps)} seconds')
else:
    print(f'Duration: {round(time_elaps/60)} minutes')