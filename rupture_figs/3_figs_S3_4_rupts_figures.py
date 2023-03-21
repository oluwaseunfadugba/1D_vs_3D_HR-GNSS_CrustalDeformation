#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 23:45:37 2023

@author: oluwaseunfadugba
"""

import  os
import matplotlib.pyplot as plt
from string import ascii_lowercase as alphab
import time
start = time.time()

current_dir = '/Users/oluwaseunfadugba/Documents/Projects/TsE_ValerieDiego/'+\
    'TsE_1D_vs_3D/1D_vs_3D_HR-GNSS_CrustalDeformation/rupture_figs/'

os.chdir(current_dir)

# create figure
fig = plt.figure(figsize=(20, 20))

# setting values to rows and column variables
rows = 3
columns = 1


#%% Figure S3
comp_all = ['Ibaraki_ZHENG','Miyagi2011_Hayes','Miyagi2011_ZHENG']

for i in range(3):
    # reading images
    Image1 = plt.imread(current_dir+"/fig.mean_n_fqs_srfs_"+comp_all[i]+".png")
    
    fig.add_subplot(rows, columns, i+1)
    plt.imshow(Image1)
    plt.axis('off')
    #plt.text(0.5, 0.5, '('+alphab[i].upper()+')', color='k', fontsize=25)#, weight='bold')
    plt.tight_layout()
      
fig.savefig('figS3.rupt_plots.png',dpi=500,\
    bbox_inches='tight',facecolor='white', edgecolor='none')

    
#%% Figure S4
comp_all = ['Iwate2011_ZHENG','tokachi2003_Hayes','tokachi2003_srcmod3']

for i in range(3):
    # reading images
    Image1 = plt.imread(current_dir+"/fig.mean_n_fqs_srfs_"+comp_all[i]+".png")
    
    fig.add_subplot(rows, columns, i+1)
    plt.imshow(Image1)
    plt.axis('off')
    #plt.text(0.5, 0.5, '('+alphab[i].upper()+')', color='k', fontsize=25)#, weight='bold')
    plt.tight_layout()
      
fig.savefig('figS4.rupt_plots.png',dpi=500,\
    bbox_inches='tight',facecolor='white', edgecolor='none')


#%% Figure 2
os.system('scp fig.mean_n_fqs_srfs_Ibaraki_SRCMOD.png fig2.rupts_Ibaraki_SRCMOD.png')

#%% ####################################################################
end = time.time()
time_elaps = end - start
if time_elaps < 60:
    print(f'Duration: {round(time_elaps)} seconds')
else:
    print(f'Duration: {round(time_elaps/60)} minutes')