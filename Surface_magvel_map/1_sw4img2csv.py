#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  2 11:26:23 2023

@author: oluwaseunfadugba
"""

import os
import time
import pySW4 as sw4
from glob import glob
import numpy as np
import pandas as pd 
start = time.time()

os.sys.path.insert(0, "/Users/oluwaseunfadugba/code/MudPy/src/python")

cwd = '/Users/oluwaseunfadugba/Documents/Projects/TsE_ValerieDiego/TsE_1D_vs_3D/'+\
    '1D_vs_3D_HR-GNSS_CrustalDeformation/Surface_magvel_map/'
os.chdir(cwd)

sw4_outpath = '/Users/oluwaseunfadugba/Documents/Projects/TsE_ValerieDiego/TsE_1D_vs_3D/'+\
    '3D_Modeling_using_SW4/Running_Simulations/4_Model_results/All_025Hz_Result/'+\
    'ibaraki2011_srcmod_srf3d_rupt5_fullrfile_talapas/ibaraki2011_srcmod_srf3d_rupt5.sw4output/'

# create folder for results
os.system('mv surf_magvel_csv surf_magvel_csv_old')
os.system('mkdir surf_magvel_csv')

# Get the file list for each time step
file_list = np.array(sorted(glob(sw4_outpath + '/*.z=0.mag.sw4img')))

# Loop overthe 
for i in range(len(file_list)):
    # Read sw4 image
    imagefile = sw4.read_image(file_list[i])

    # Extract cycle number and convert it to time stamp
    clcle_no = file_list[i].split('.')[-4][-5:]
    t=str(round(float(clcle_no)*0.0219623)-40)
    
    # Print progress
    print(clcle_no,t)
    
    # Get surface magvel data and print it to csv file
    patch=imagefile.patches[0]
    matrix=patch.data
    pd.DataFrame(matrix[::-1]).to_csv('surf_magvel_csv/surf_magvel_at_time_'+
                                      t+'s.csv', header=None, index=None)

# ####################################################################
end = time.time()
time_elaps = end - start
if time_elaps < 60:
    print(f'Duration: {round(time_elaps)} seconds')
else:
    print(f'Duration: {round(time_elaps/60)} minutes')