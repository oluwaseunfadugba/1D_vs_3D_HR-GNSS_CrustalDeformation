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


#%%
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


#%% create figure
fig, axs = plt.subplots(3,4,figsize=(32,27), sharey='row', sharex='col')
# fig.subplots_adjust(wspace=0.5,hspace=0.1)
#plt.subplots_adjust(wspace = 0.2)# ,hspace = 0.1, right=0.1)
#fig.tight_layout(pad=0.4, w_pad=0.5, h_pad=7.0,rect=[0, 0.15, 0.8, 0.9]) #0, 0.03, 0.85, 0.90]
axs=axs.flatten()
fontsize = 35

fig.suptitle('Surface Magnitude Velocity for Ibaraki 2011 \n (SRCMOD Rupture 5)', fontsize=45)

im1 = plt.imread(outputfilename)
for i in range(len(t)):
    print(i)

    axs[i].imshow(im1, extent=region,alpha=1)

    stadata = pd.read_csv('surf_magvel_csv/surf_magvel_at_time_'+str(t[i])+'s.csv')
    im=axs[i].imshow(stadata,extent=region, cmap='whitejet_alpha', vmin=0, vmax=0.08,alpha=6)
    
    axs[i].set_aspect(1.5)
    if i >= 8:
        axs[i].set_xlabel('longitude',fontsize=fontsize+5)
    if i == 0 or i == 4 or i == 8:
        axs[i].set_ylabel('latitude',fontsize=fontsize+5)
        axs[i].text(125, 45.2, '('+alphab[i].upper()+')', color='k', fontsize=fontsize+5)#, weight='bold')
    else:
        axs[i].text(127, 45.2, '('+alphab[i].upper()+')', color='k', fontsize=fontsize+5)#, weight='bold')
    axs[i].set_xticks([135,140,145])
    axs[i].set_xticklabels([135,140,145],rotation=45, ha='right')
    axs[i].set_yticks([35,40,45])
    axs[i].tick_params(labelsize=fontsize+5)
    axs[i].set_title('time = '+str(t[i])+' s',fontsize=fontsize+5)

cbar_ax = fig.add_axes([0.25, -0.00, 0.5, 0.025])
cbar=fig.colorbar(im, cax=cbar_ax,orientation="horizontal")
cbar.set_label('Surface Mag Velocity',fontsize=fontsize+5)
cbar.ax.tick_params(labelsize=fontsize)

#plt.show()
fig.savefig('fig8_surf_magvel_snapshots.png',dpi=200,\
    bbox_inches='tight',facecolor='white', edgecolor='none')


# ####################################################################
end = time.time()
time_elaps = end - start
if time_elaps < 60:
    print(f'Duration: {round(time_elaps)} seconds')
else:
    print(f'Duration: {round(time_elaps/60)} minutes')