#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 12:22:01 2023

@author: oluwaseunfadugba
"""
# This function can only run under the mtspec environment
# To activate, close the Jupyter Notebook, run the command below and 
# restart the Jupyter Notebook
#     'conda activate mtspec'
# # For talapas, I have to remove 20 s from the start and end time. It's a mistake in the srf

import os

path = '/Users/oluwaseunfadugba/Documents/Projects/TsE_ValerieDiego/TsE_1D_vs_3D/'+\
    '3D_Modeling_using_SW4/Running_Simulations/4_Model_results/'
os.chdir(path)

import numpy as np
import time
import cal_IM_residuals_3D_now_funcs as funcs
start = time.time()

gflist = path + 'ALL_events.GNSS_locs.txt'


#%% ibaraki2011_srcmod_srf3d full rfile (3D z30 km)
eqname = 'ibaraki2011_srcmod_srf3d_z30km_talapas'; wfs = 'All_025Hz_Result/'; fmax= 0.25
obs_wfs = '/Users/oluwaseunfadugba/Documents/Projects/TsE_ValerieDiego/Data_GNSSdata/Ibaraki2011/disp'
eqparam = ["2011-03-11T06:15:34",141.2653,36.1083,43.2,7.9,1,fmax]
flatfile = wfs + 'flatfile_'+eqname+'_'+str(fmax)+'Hz_Residuals.csv'

data =np.array([['ibaraki2011_srcmod_srf3d_z30km_rupt5_talapas',5,1], # folder_name, rupt number, lasen/talapas
                ['ibaraki2011_srcmod_srf3d_z30km_rupt9_talapas',9,1]],dtype=object) 

funcs.process_IM_residuals(eqparam,path,gflist,eqname,obs_wfs,wfs,flatfile,data)
print(' ')




#%% ibaraki2011_srcmod_srf3d full rfile
# eqname = 'ibaraki2011_srcmod_srf3d_rupt5_fullrfile_talapas'; wfs = 'All_025Hz_Result/'; fmax= 0.25
# obs_wfs = '/Users/oluwaseunfadugba/Documents/Projects/TsE_ValerieDiego/Data_GNSSdata/Ibaraki2011/disp'
# eqparam = ["2011-03-11T06:15:34",141.2653,36.1083,43.2,7.9,1,fmax]
# flatfile = wfs + 'flatfile_'+eqname+'_'+str(fmax)+'Hz_Residuals.csv'

# data =np.array([#['ibaraki2011_srcmod_srf3d_rupt5_talapas',5,1], # folder_name, rupt number, lasen/talapas
#                 ['ibaraki2011_srcmod_srf3d_rupt5_fullrfile_talapas',5,1]],dtype=object) 

# funcs.process_IM_residuals(eqparam,path,gflist,eqname,obs_wfs,wfs,flatfile,data)
# print(' ')




#%% ibaraki2011_srcmod_srf3d

# eqname = 'ibaraki2011_srcmod_srf3d_talapas'; wfs = 'All_025Hz_Result/'; fmax= 0.25
# obs_wfs = '/Users/oluwaseunfadugba/Documents/Projects/TsE_ValerieDiego/Data_GNSSdata/Ibaraki2011/disp'
# eqparam = ["2011-03-11T06:15:34",141.2653,36.1083,43.2,7.9,1,fmax]
# flatfile = wfs + 'flatfile_'+eqname+'_'+str(fmax)+'Hz_Residuals.csv'

# data =np.array([['ibaraki2011_srcmod_srf3d_rupt5_talapas',5,1], # folder_name, rupt number, lasen/talapas
#                 ['ibaraki2011_srcmod_srf3d_rupt9_talapas',9,1]],dtype=object) 

# funcs.process_IM_residuals(eqparam,path,gflist,eqname,obs_wfs,wfs,flatfile,data)
# print(' ')

# eqname = 'ibaraki2011_srcmod_srf3d_lasson'; wfs = 'All_025Hz_Result/'; fmax= 0.25
# obs_wfs = '/Users/oluwaseunfadugba/Documents/Projects/TsE_ValerieDiego/Data_GNSSdata/Ibaraki2011/disp'
# eqparam = ["2011-03-11T06:15:34",141.2653,36.1083,43.2,7.9,1,fmax]
# flatfile = wfs + 'flatfile_'+eqname+'_'+str(fmax)+'Hz_Residuals.csv'

# data =np.array([['ibaraki2011_srcmod_srf3d_rupt1',0,0], # folder_name, rupt number, lasen/talapas
#                 ['ibaraki2011_srcmod_srf3d_rupt2',1,0]],dtype=object)   

# funcs.process_IM_residuals(eqparam,path,gflist,eqname,obs_wfs,wfs,flatfile,data)
# print(' ')

# #%% ibaraki2011_zheng1_srf3d

# eqname = 'ibaraki2011_zheng1_srf3d_talapas'; wfs = 'All_025Hz_Result/'; fmax= 0.25
# obs_wfs = '/Users/oluwaseunfadugba/Documents/Projects/TsE_ValerieDiego/Data_GNSSdata/Ibaraki2011/disp'
# eqparam = ["2011-03-11T06:15:34",141.2653,36.1083,43.2,7.9,1,fmax]
# flatfile = wfs + 'flatfile_'+eqname+'_'+str(fmax)+'Hz_Residuals.csv'

# data =np.array([['ibaraki2011_zheng1_srf3d_rupt0_talapas',0,1], # folder_name, rupt number, lasen/talapas
#                 ['ibaraki2011_zheng1_srf3d_rupt1_talapas',1,1]],dtype=object)   

# funcs.process_IM_residuals(eqparam,path,gflist,eqname,obs_wfs,wfs,flatfile,data)
# print(' ')

# eqname = 'ibaraki2011_zheng1_srf3d_lasson'; wfs = 'All_025Hz_Result/'; fmax= 0.25
# obs_wfs = '/Users/oluwaseunfadugba/Documents/Projects/TsE_ValerieDiego/Data_GNSSdata/Ibaraki2011/disp'
# eqparam = ["2011-03-11T06:15:34",141.2653,36.1083,43.2,7.9,1,fmax]
# flatfile = wfs + 'flatfile_'+eqname+'_'+str(fmax)+'Hz_Residuals.csv'

# data =np.array([['ibaraki2011_zheng1_srf3d_rupt1',0,0], # folder_name, rupt number, lasen/talapas
#                 ['ibaraki2011_zheng1_srf3d_rupt2',1,0]],dtype=object)   

# funcs.process_IM_residuals(eqparam,path,gflist,eqname,obs_wfs,wfs,flatfile,data)
# print(' ')

# #%% iwate2011_zheng1_srf3d

# eqname = 'iwate2011_zheng1_srf3d_talapas'; wfs = 'All_025Hz_Result/'; fmax= 0.25
# obs_wfs = '/Users/oluwaseunfadugba/Documents/Projects/TsE_ValerieDiego/Data_GNSSdata/Iwate2011/disp'
# eqparam = ["2011-03-11T06:08:53",142.7815,39.8390,31.7,7.4,1,fmax]
# flatfile = wfs + 'flatfile_'+eqname+'_'+str(fmax)+'Hz_Residuals.csv'

# data =np.array([['iwate2011_zheng1_srf3d_rupt0_talapas',0,1], # folder_name, rupt number, lasen/talapas
#                 ['iwate2011_zheng1_srf3d_rupt3_talapas',3,1]],dtype=object)   

# funcs.process_IM_residuals(eqparam,path,gflist,eqname,obs_wfs,wfs,flatfile,data)
# print(' ')

# eqname = 'iwate2011_zheng1_srf3d_lasson'; wfs = 'All_025Hz_Result/'; fmax= 0.25
# obs_wfs = '/Users/oluwaseunfadugba/Documents/Projects/TsE_ValerieDiego/Data_GNSSdata/Iwate2011/disp'
# eqparam = ["2011-03-11T06:08:53",142.7815,39.8390,31.7,7.4,1,fmax]
# flatfile = wfs + 'flatfile_'+eqname+'_'+str(fmax)+'Hz_Residuals.csv'

# data =np.array([['iwate2011_zheng1_srf3d_rupt1',0,0], # folder_name, rupt number, lasen/talapas
#                 ['iwate2011_zheng1_srf3d_rupt2',1,0]],dtype=object)   

# funcs.process_IM_residuals(eqparam,path,gflist,eqname,obs_wfs,wfs,flatfile,data)
# print(' ')


# #%% miyagi2011a_zheng1_srf3d

# eqname = 'miyagi2011a_zheng1_srf3d_talapas'; wfs = 'All_025Hz_Result/'; fmax= 0.25
# obs_wfs = '/Users/oluwaseunfadugba/Documents/Projects/TsE_ValerieDiego/Data_GNSSdata/Miyagi2011A/disp'
# eqparam = ["2011-03-09T02:45:12",143.2798,38.3285,8.3,7.3,1,fmax]
# flatfile = wfs + 'flatfile_'+eqname+'_'+str(fmax)+'Hz_Residuals.csv'

# data =np.array([['miyagi2011a_zheng1_srf3d_rupt0_talapas',0,1], # folder_name, rupt number, lasen/talapas
#                 ['miyagi2011a_zheng1_srf3d_rupt3_talapas',3,1]],dtype=object)   

# funcs.process_IM_residuals(eqparam,path,gflist,eqname,obs_wfs,wfs,flatfile,data)
# print(' ')


# eqname = 'miyagi2011a_zheng1_srf3d_lasson'; wfs = 'All_025Hz_Result/'; fmax= 0.25
# obs_wfs = '/Users/oluwaseunfadugba/Documents/Projects/TsE_ValerieDiego/Data_GNSSdata/Miyagi2011A/disp'
# eqparam = ["2011-03-09T02:45:12",143.2798,38.3285,8.3,7.3,1,fmax]
# flatfile = wfs + 'flatfile_'+eqname+'_'+str(fmax)+'Hz_Residuals.csv'

# data =np.array([['miyagi2011a_zheng1_srf3d_rupt1',0,0], # folder_name, rupt number, lasen/talapas
#                 ['miyagi2011a_zheng1_srf3d_rupt2',1,0]],dtype=object)   

# funcs.process_IM_residuals(eqparam,path,gflist,eqname,obs_wfs,wfs,flatfile,data)
# print(' ')




# #%% miyagi2011a_usgs_srf3d

# eqname = 'miyagi2011a_usgs_srf3d_talapas'; wfs = 'All_025Hz_Result/'; fmax= 0.25
# obs_wfs = '/Users/oluwaseunfadugba/Documents/Projects/TsE_ValerieDiego/Data_GNSSdata/Miyagi2011A/disp'
# eqparam = ["2011-03-09T02:45:12",143.2798,38.3285,8.3,7.3,1,fmax]
# flatfile = wfs + 'flatfile_'+eqname+'_'+str(fmax)+'Hz_Residuals.csv'

# data =np.array([['miyagi2011a_usgs_srf3d_rupt1_talapas',1,1], # folder_name, rupt number, lasen/talapas
#                 ['miyagi2011a_usgs_srf3d_rupt2_talapas',2,1]],dtype=object)   

# funcs.process_IM_residuals(eqparam,path,gflist,eqname,obs_wfs,wfs,flatfile,data)
# print(' ')


# eqname = 'miyagi2011a_usgs_srf3d_lasson'; wfs = 'All_025Hz_Result/'; fmax= 0.25
# obs_wfs = '/Users/oluwaseunfadugba/Documents/Projects/TsE_ValerieDiego/Data_GNSSdata/Miyagi2011A/disp'
# eqparam = ["2011-03-09T02:45:12",143.2798,38.3285,8.3,7.3,1,fmax]
# flatfile = wfs + 'flatfile_'+eqname+'_'+str(fmax)+'Hz_Residuals.csv'

# data =np.array([['miyagi2011a_usgs_srf3d_rupt1',0,0], # folder_name, rupt number, lasen/talapas         # was commented out before
#                 ['miyagi2011a_usgs_srf3d_rupt2',1,0]],dtype=object)   

# funcs.process_IM_residuals(eqparam,path,gflist,eqname,obs_wfs,wfs,flatfile,data)
# print(' ')


# #%% miyagi2011a_zheng1_srf3d_z30km

# eqname = 'miyagi2011a_zheng1_srf3d_z30km_talapas'; wfs = 'All_025Hz_Result/'; fmax= 0.25
# obs_wfs = '/Users/oluwaseunfadugba/Documents/Projects/TsE_ValerieDiego/Data_GNSSdata/Miyagi2011A/disp'
# eqparam = ["2011-03-09T02:45:12",143.2798,38.3285,8.3,7.3,1,fmax]
# flatfile = wfs + 'flatfile_'+eqname+'_'+str(fmax)+'Hz_Residuals.csv'

# data =np.array([['miyagi2011a_zheng1_srf3d_z30km_rupt0_talapas',0,1], # folder_name, rupt number, lasen/talapas
#                 ['miyagi2011a_zheng1_srf3d_z30km_rupt3_talapas',3,1]],dtype=object)   

# funcs.process_IM_residuals(eqparam,path,gflist,eqname,obs_wfs,wfs,flatfile,data)
# print(' ')



# #%% tokachi2003_usgs_srf3d

# eqname = 'tokachi2003_usgs_srf3d_talapas'; wfs = 'All_025Hz_Result/'; fmax= 0.25
# obs_wfs = '/Users/oluwaseunfadugba/Documents/Projects/TsE_ValerieDiego/Data_GNSSdata/Tokachi2003/disp'
# eqparam = ["2003-09-25T19:50:06",143.9040,41.7750,27.0,8.3,1,fmax]
# flatfile = wfs + 'flatfile_'+eqname+'_'+str(fmax)+'Hz_Residuals.csv'

# data =np.array([['tokachi2003_usgs_srf3d_rupt0_talapas',0,1], # folder_name, rupt number, lasen/talapas
#                 ['tokachi2003_usgs_srf3d_rupt1_talapas',1,1]],dtype=object)   

# funcs.process_IM_residuals(eqparam,path,gflist,eqname,obs_wfs,wfs,flatfile,data)
# print(' ')


# eqname = 'tokachi2003_usgs_srf3d_lasson'; wfs = 'All_025Hz_Result/'; fmax= 0.25
# obs_wfs = '/Users/oluwaseunfadugba/Documents/Projects/TsE_ValerieDiego/Data_GNSSdata/Tokachi2003/disp'
# eqparam = ["2003-09-25T19:50:06",143.9040,41.7750,27.0,8.3,1,fmax]
# flatfile = wfs + 'flatfile_'+eqname+'_'+str(fmax)+'Hz_Residuals.csv'

# data =np.array([['tokachi2003_usgs_srf3d_rupt1',0,0], # folder_name, rupt number, lasen/talapas
#                 ['tokachi2003_usgs_srf3d_rupt2',1,0]],dtype=object)   

# funcs.process_IM_residuals(eqparam,path,gflist,eqname,obs_wfs,wfs,flatfile,data)
# print(' ')


# #%% tokachi2003_srcmod3_srf3d

# eqname = 'tokachi2003_srcmod3_srf3d_talapas'; wfs = 'All_025Hz_Result/'; fmax= 0.25
# obs_wfs = '/Users/oluwaseunfadugba/Documents/Projects/TsE_ValerieDiego/Data_GNSSdata/Tokachi2003/disp'
# eqparam = ["2003-09-25T19:50:06",143.9040,41.7750,27.0,8.3,1,fmax]
# flatfile = wfs + 'flatfile_'+eqname+'_'+str(fmax)+'Hz_Residuals.csv'

# data =np.array([['tokachi2003_srcmod3_srf3d_rupt1_talapas',1,1], # folder_name, rupt number, lasen/talapas
#                 ['tokachi2003_srcmod3_srf3d_rupt4_talapas',4,1]],dtype=object)   

# funcs.process_IM_residuals(eqparam,path,gflist,eqname,obs_wfs,wfs,flatfile,data)
# print(' ')


# #%% 0.5 Hz results
# #%% ibaraki2011_srcmod_srf3d

# eqname = 'ibaraki2011_srcmod_srf3d_talapas'; wfs = 'All_05Hz_Results/'; fmax= 0.49
# obs_wfs = '/Users/oluwaseunfadugba/Documents/Projects/TsE_ValerieDiego/Data_GNSSdata/Ibaraki2011/disp'
# eqparam = ["2011-03-11T06:15:34",141.2653,36.1083,43.2,7.9,1,fmax]
# flatfile = wfs + 'flatfile_'+eqname+'_'+str(fmax)+'Hz_Residuals.csv'

# data =np.array([['ibaraki2011_scrmod_srf3d_rupt6_long',6,1], # folder_name, rupt number, lasen/talapas
#                 ['ibaraki2011_srcmod_srf3d_rupt5',5,1]],dtype=object) 

# funcs.process_IM_residuals(eqparam,path,gflist,eqname,obs_wfs,wfs,flatfile,data)
# print(' ')



# #%% ibaraki2011_zheng1_srf3d
   
# eqname = 'ibaraki2011_zheng1_srf3d_talapas'; wfs = 'All_05Hz_Results/'; fmax= 0.49
# obs_wfs = '/Users/oluwaseunfadugba/Documents/Projects/TsE_ValerieDiego/Data_GNSSdata/Ibaraki2011/disp'
# eqparam = ["2011-03-11T06:15:34",141.2653,36.1083,43.2,7.9,1,fmax]
# flatfile = wfs + 'flatfile_'+eqname+'_'+str(fmax)+'Hz_Residuals.csv'

# data =np.array([
#                 #[], # folder_name, rupt number, lasen/talapas
#                 ['ibaraki2011_zheng1_srf3d_rupt1_long',1,1]],dtype=object) 

# funcs.process_IM_residuals(eqparam,path,gflist,eqname,obs_wfs,wfs,flatfile,data)
# print(' ')


# #%% iwate2011_zheng1_srf3d

# eqname = 'iwate2011_zheng1_srf3d_talapas'; wfs = 'All_05Hz_Results/'; fmax= 0.49
# obs_wfs = '/Users/oluwaseunfadugba/Documents/Projects/TsE_ValerieDiego/Data_GNSSdata/Iwate2011/disp'
# eqparam = ["2011-03-11T06:08:53",142.7815,39.8390,31.7,7.4,1,fmax]
# flatfile = wfs + 'flatfile_'+eqname+'_'+str(fmax)+'Hz_Residuals.csv'

# data =np.array([
#                 #[',0,1], # folder_name, rupt number, lasen/talapas
#                 ['iwate2011_zheng1_srf3d_rupt0',0,1]],dtype=object)   

# funcs.process_IM_residuals(eqparam,path,gflist,eqname,obs_wfs,wfs,flatfile,data)
# print(' ')

# #%% miyagi2011a_zheng1_srf3d

# eqname = 'miyagi2011a_zheng1_srf3d_talapas'; wfs = 'All_05Hz_Results/'; fmax= 0.49
# obs_wfs = '/Users/oluwaseunfadugba/Documents/Projects/TsE_ValerieDiego/Data_GNSSdata/Miyagi2011A/disp'
# eqparam = ["2011-03-09T02:45:12",143.2798,38.3285,8.3,7.3,1,fmax]
# flatfile = wfs + 'flatfile_'+eqname+'_'+str(fmax)+'Hz_Residuals.csv'

# data =np.array([
#                 #[], # folder_name, rupt number, lasen/talapas
#                 ['miyagi2011_zheng1_srf3d_rupt0',0,1]],dtype=object)   

# funcs.process_IM_residuals(eqparam,path,gflist,eqname,obs_wfs,wfs,flatfile,data)
# print(' ')

# #%% miyagi2011a_usgs_srf3d

# eqname = 'miyagi2011a_usgs_srf3d_talapas'; wfs = 'All_05Hz_Results/'; fmax= 0.49
# obs_wfs = '/Users/oluwaseunfadugba/Documents/Projects/TsE_ValerieDiego/Data_GNSSdata/Miyagi2011A/disp'
# eqparam = ["2011-03-09T02:45:12",143.2798,38.3285,8.3,7.3,1,fmax]
# flatfile = wfs + 'flatfile_'+eqname+'_'+str(fmax)+'Hz_Residuals.csv'

# data =np.array([
#                 #[',0,1], # folder_name, rupt number, lasen/talapas
#                 ['miyagi2011a_usgs_srf3d_rupt1',1,1]],dtype=object)   

# funcs.process_IM_residuals(eqparam,path,gflist,eqname,obs_wfs,wfs,flatfile,data)
# print(' ')

# #%% tokachi2003_srcmod3_srf3d

# eqname = 'tokachi2003_srcmod3_srf3d_talapas'; wfs = 'All_05Hz_Results/'; fmax= 0.49
# obs_wfs = '/Users/oluwaseunfadugba/Documents/Projects/TsE_ValerieDiego/Data_GNSSdata/Tokachi2003/disp'
# eqparam = ["2003-09-25T19:50:06",143.9040,41.7750,27.0,8.3,1,fmax]
# flatfile = wfs + 'flatfile_'+eqname+'_'+str(fmax)+'Hz_Residuals.csv'

# data =np.array([
#                 #['',1,1], # folder_name, rupt number, lasen/talapas
#                 ['tokachi2003_srcmod3_srf3d_rupt1',1,1]],dtype=object)   

# funcs.process_IM_residuals(eqparam,path,gflist,eqname,obs_wfs,wfs,flatfile,data)
# print(' ')


# #%% tokachi2003_srcmod3_srf3d

# eqname = 'tokachi2003_usgs_srf3d_talapas'; wfs = 'All_05Hz_Results/'; fmax= 0.49
# obs_wfs = '/Users/oluwaseunfadugba/Documents/Projects/TsE_ValerieDiego/Data_GNSSdata/Tokachi2003/disp'
# eqparam = ["2003-09-25T19:50:06",143.9040,41.7750,27.0,8.3,1,fmax]
# flatfile = wfs + 'flatfile_'+eqname+'_'+str(fmax)+'Hz_Residuals.csv'

# data =np.array([
#                 #['',1,1], # folder_name, rupt number, lasen/talapas
#                 ['tokachi2003_usgs_srf3d_rupt0',0,1]],dtype=object)   

# funcs.process_IM_residuals(eqparam,path,gflist,eqname,obs_wfs,wfs,flatfile,data)
# print(' ')



# ####################################################################
end = time.time()
time_elaps = end - start
if time_elaps < 60:
    print(f'Duration: {round(time_elaps)} seconds')
else:
    print(f'Duration: {round(time_elaps/60)} minutes')

      
#ibaraki - 700/728
#iwate - good
#miyagi - 163/198
#tokachi - 121/236
