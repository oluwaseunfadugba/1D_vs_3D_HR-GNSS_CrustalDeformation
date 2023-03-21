#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  3 13:40:32 2023

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

# create figure
fig = plt.figure(figsize=(20, 20))

# setting values to rows and column variables
rows = 2
columns = 2

comp_all = ['pgd_res','tPGD_res','sd_res','xcorr']

for i in range(4):
    # reading images
    Image1 = plt.imread(current_dir+"/fig."+comp_all[i]+"_ibaraki_srcmod.png")
    
    fig.add_subplot(rows, columns, i+1)
    plt.imshow(Image1)
    plt.axis('off')
    plt.text(0.5, 0.5, '('+alphab[i].upper()+')', color='k', fontsize=25)#, weight='bold')
    plt.tight_layout()
      
fig.savefig('figS7.residual_maps_ibaraki2011.png',dpi=500,\
    bbox_inches='tight',facecolor='white', edgecolor='none')

# ####################################################################
end = time.time()
time_elaps = end - start
if time_elaps < 60:
    print(f'Duration: {round(time_elaps)} seconds')
else:
    print(f'Duration: {round(time_elaps/60)} minutes')