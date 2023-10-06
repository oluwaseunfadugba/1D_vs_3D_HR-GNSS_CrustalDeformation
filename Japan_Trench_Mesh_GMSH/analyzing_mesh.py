#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 18 14:29:51 2022

@author: oluwaseunfadugba
"""

import os
import numpy as np
from glob import glob
from pathlib import Path
import time
import matplotlib.pyplot as plt

start = time.time()


home_dir = '/Users/oluwaseunfadugba/Documents/Projects/TsE_ValerieDiego/'+\
    'TsE_1D_vs_3D/1D_Modeling_using_FQs_Mudpy/Creating_Inputs/Japan_Trench_Mesh_GMSH/'
os.chdir(home_dir)

input_file = home_dir + '/Japan_trench.fault'


def histo(ax,data,title,xlabel):
    ax.hist(data)#, bins = [0, 1000, 2000, 3000,4000, 5000,
                        #6000, 7000,8000, 9000, 10000,11000,12000])
    
    label_font = 20
    ax.set_title(title, fontsize=label_font+3)
    ax.set_xlabel(xlabel, fontsize=label_font)
    ax.set_ylabel('freq', fontsize=label_font)
    ax.tick_params(axis='x',labelsize=label_font,labelrotation=0)
    ax.tick_params(axis='y',labelsize=label_font,labelrotation=0)
    
    ax.grid()
    
#%%
data = np.genfromtxt(input_file)  

lon = data[:,1]
lat = data[:,2]
depth = data[:,3]
strike = data[:,4]
dip = data[:,5]
length = data[:,8]
width = data[:,9]



print('lat',lat.min(),lat.max())
print('lon',lon.min(),lat.max())
print('depth',depth.min(),depth.max())
print('width',width.min(),width.max(),width.mean())


# fig,ax = plt.subplots(2,3,figsize =(25, 18))
# fig.suptitle('Histogram of Epicentral Distance',fontsize=30)


# histo(ax[0,0],width,'Wdith/length','width (km)')
# histo(ax[0,1],strike,'strike','strike (deg)')
# histo(ax[0,2],dip,'dip','dip (deg)')

