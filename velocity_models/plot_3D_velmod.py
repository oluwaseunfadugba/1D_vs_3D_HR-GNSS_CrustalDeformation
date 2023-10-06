#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 13:03:16 2023

@author: oluwaseunfadugba
"""
import matplotlib.pyplot as plt
import sys
import os

flush = sys.stdout.flush()

from pySW4.prep import rfileIO
import numpy as np
import time
import matplotlib.ticker as ticker
start = time.time()

outpath = '/Users/oluwaseunfadugba/Documents/Projects/TsE_ValerieDiego/TsE_1D_vs_3D/'+\
    '1D_vs_3D_HR-GNSS_CrustalDeformation/velocity_models/'

os.chdir(outpath)

rfile = 'Creating_rfile_5nblks_z200_h500/3djapan_hh=1000m_hv=2_3_4_500m.rfile'

#%% Reading and plotting the rfile
model = rfileIO.read(rfile, 'all', verbose=True)

#%% Plotting the Cross-sections
fig, axs = plt.subplots(1,2,figsize=(12,30))
plt.subplots_adjust(wspace = 0.2)# ,hspace = 0.1, right=0.1)
axs=axs.flatten()
fontsize = 35

cs = model.get_cross_section(0, 2040, 40, 40)
fig, ax, cb = cs.plot(ax=axs[0],property='vp', vmin=1800,aspect=7.5, cmap='jet') #vmin=0,
ax.set_title('')
ax.tick_params(labelsize=16)
ax.set_xlabel('Distance from origin [km]', fontsize=16)
ax.set_ylabel('Depth from sea-level [km]', fontsize=16)
ax.text(-100, -15, 'A', color='k', fontsize=20)
ax.text(2000, -15, 'B', color='k', fontsize=20)
cb.remove()

#------------------------------------------------------------------------------
cs2 = model.get_cross_section(1500, 1500, 0, 1400)
fig, ax, cb = cs2.plot(ax=axs[1],property='vp', vmin=1800, aspect=5, cmap='jet')
ax.set_title('')
ax.text(-100, -15, 'C', color='k', fontsize=20)
ax.text(1350, -15, 'D', color='k', fontsize=20)
ax.tick_params(labelsize=16)
ax.set_xlabel('Distance from origin [km]', fontsize=16)
ax.set_ylabel('')

# Define a formatter to display integer tick labels
formatter = ticker.FuncFormatter(lambda x, pos: '{:.1f}'.format(x/1000))
cb.ax.yaxis.set_major_formatter(formatter)
cb.ax.set_yticklabels(np.arange(1., 9.),fontsize=16)
                        
cb.set_label('Vp [km/s]', labelpad=10, fontsize=16)

plt.rcParams.update({'font.size': 20})

figpath = os.getcwd() +'/fig4.3d_vel_crossection.png'
plt.savefig(figpath, bbox_inches='tight', dpi=200) 
plt.show()


# #%% Plotting vs profiles
# fig, axs = plt.subplots(1,2,figsize=(20,10))
# plt.subplots_adjust(wspace = 0.3)# ,hspace = 0.1, right=0.1)
# axs=axs.flatten()
# fontsize = 35

# z, properties = model.get_z_profile(10, 10)

# linewidth = 5
# fontsize = 30

# rho = np.ma.masked_equal(properties.T[0],-999)
# vp = np.ma.masked_equal(properties.T[1],-999)
# vs = np.ma.masked_equal(properties.T[2],-999)
# qp = np.ma.masked_equal(properties.T[3],-999)
# qs = np.ma.masked_equal(properties.T[4],-999)

# axs[0].plot(vp,z, 'r--',label="Vp (m/s)",linewidth=linewidth)
# axs[0].plot(vs, z, 'b-.',label="Vs (m/s)",linewidth=linewidth)
# axs[0].plot(rho, z, 'g-',label="den (g/cm3)",linewidth=linewidth)
# axs[0].plot(qs, z, 'c-',label="Qs",linewidth=linewidth)
# axs[0].plot(qp, z, 'm--',label="Qp",linewidth=linewidth)

# axs[0].set_title('Profiles at location (10,10)',fontsize=40, pad=50)
# axs[0].set_xlabel('Material properties',fontsize=fontsize)
# axs[0].set_ylabel('depth (km)',fontsize=fontsize)

# axs[0].xaxis.set_tick_params(labelsize=fontsize)
# axs[0].yaxis.set_tick_params(labelsize=fontsize)

# axs[0].set_ylim((0,z[-1]))
# lgd = axs[0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
#       fancybox=False, shadow=True, ncol=3,fontsize=25) #, 
# axs[0].grid(color='k', linestyle='-', linewidth=0.5)
# axs[0].invert_yaxis()
# #plt.xaxis.set_label_position('top')

# #------------------------------------------------------------------------------
# # Vs profile at different locations (x,y) within the domain
# lat_bin = np.linspace(0.0, 2040, 601)
# lon_bin = np.linspace(0.0, 1400, 401)

# for i in range(len(lat_bin)):
#     for j in range(len(lon_bin)):
#         z, properties = model.get_z_profile(lon_bin[j], lat_bin[i])

#         p = np.ma.masked_equal(properties.T[2], -999)
#         try:
#             axs[1].plot(p, z,linewidth=0.5) 
#         except:
#             continue
        
# axs[1].set_title('Vs Profiles at several locations',fontsize=40, pad=50)
# axs[1].set_xlabel('Vs (m/s)',fontsize=fontsize)
# axs[1].set_ylabel('depth (km)',fontsize=fontsize)

# axs[1].xaxis.set_tick_params(labelsize=fontsize)
# axs[1].yaxis.set_tick_params(labelsize=fontsize)

# axs[1].set_ylim((0,z[-1]))
# axs[1].grid(color='k', linestyle='-', linewidth=0.5)
# axs[1].invert_yaxis()


# figpath = os.getcwd() +'/figS6.mat_profiles_rfile.png'
# plt.savefig(figpath, bbox_inches='tight', dpi=200)    
# plt.show()

# #%% Plot_topography
# model.plot_topography(cmap='terrain')
# figpath = os.getcwd() +'/fig.rfiletopo.png'
# plt.savefig(figpath, bbox_inches='tight', dpi=200) 
# plt.show()

# #%% Plot model.get_cross_section
# cs = model.get_cross_section(0, 2040, 400, 400)
# fig, ax, cb = cs.plot(property='vp', vmin=1800,aspect=10, cmap='jet') #vmin=0,
# figpath = os.getcwd() +'/fig.crossection1.png'
# plt.savefig(figpath, bbox_inches='tight', dpi=200) 
# plt.show()

# cs = model.get_cross_section(1500, 1500, 0, 1400)
# fig, ax, cb = cs.plot(property='vp', vmin=1800, aspect=5, cmap='jet')
# figpath = os.getcwd() +'/fig.crossection2.png'
# plt.savefig(figpath, bbox_inches='tight', dpi=200) 
# plt.show()

# ####################################################################
end = time.time()
time_elaps = end - start
if time_elaps < 60:
    print(f'Duration: {round(time_elaps)} seconds')
else:
    print(f'Duration: {round(time_elaps/60)} minutes')  
  