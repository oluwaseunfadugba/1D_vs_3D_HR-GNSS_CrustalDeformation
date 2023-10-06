#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 11:57:04 2023

@author: oluwaseunfadugba
"""

import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

start = time.time()

import os
import comparing_funcs as funcs


tcb10 = ['#006BA4', '#FF800E', '#ABABAB', '#595959', '#5F9ED1', '#C85200', '#898989', '#A2C8EC', '#FFBC79', '#CFCFCF']

# Color codes
# SRCMOD - 0 'o'
# ZHENG - 1   'x'
# USGS - 2   '//'
# SRCMOD2 - 3 '+'
# SRCMOD3 - 4 '<'

#%% setting flatfile paths
# 1D 0.25 Hz and 0.5 Hz
home_1D = '/Users/oluwaseunfadugba/Documents/Projects/TsE_ValerieDiego/TsE_1D_vs_3D/1D_Modeling_using_FQs_Mudpy/Running_FakeQuakes_now/'

iba_src_1d_025Hz = home_1D+'flatfile_1d_ibaraki2011_srcmod_0.25Hz_Residuals.csv'
iba_zh_1d_025Hz = home_1D+'flatfile_1d_ibaraki2011_zheng1_0.25Hz_Residuals.csv'

iwa_zh_1d_025Hz = home_1D+'flatfile_1d_iwate2011_zheng1_0.25Hz_Residuals.csv'

miy_usgs_1d_025Hz = home_1D+'flatfile_1d_miyagi2011a_usgs_0.25Hz_Residuals.csv'
miy_zh_1d_025Hz = home_1D+'flatfile_1d_miyagi2011a_zheng1_0.25Hz_Residuals.csv'

tok_src1_1d_025Hz = home_1D+'flatfile_1d_tokachi2003_srcmod1_0.25Hz_Residuals.csv'
tok_src2_1d_025Hz = home_1D+'flatfile_1d_tokachi2003_srcmod2_0.25Hz_Residuals.csv'
tok_src3_1d_025Hz = home_1D+'flatfile_1d_tokachi2003_srcmod3_0.25Hz_Residuals.csv'
tok_usgs_1d_025Hz = home_1D+'flatfile_1d_tokachi2003_usgs_0.25Hz_Residuals.csv'

# 3D 0.25Hz
home_3D_025Hz = "/Users/oluwaseunfadugba/Documents/Projects/TsE_ValerieDiego/TsE_1D_vs_3D/3D_Modeling_using_SW4/Running_Simulations/4_Model_results/All_025Hz_Result/"

iba_src_3d_025Hz_las = home_3D_025Hz + 'flatfile_ibaraki2011_srcmod_srf3d_lasson_0.25Hz_Residuals.csv'
iba_src_3d_025Hz_tal = home_3D_025Hz + 'flatfile_ibaraki2011_srcmod_srf3d_talapas_0.25Hz_Residuals.csv'

iba_zh_3d_025Hz_las = home_3D_025Hz + 'flatfile_ibaraki2011_zheng1_srf3d_lasson_0.25Hz_Residuals.csv'
iba_zh_3d_025Hz_tal = home_3D_025Hz + 'flatfile_ibaraki2011_zheng1_srf3d_talapas_0.25Hz_Residuals.csv'

iwa_zh_3d_025Hz_las = home_3D_025Hz + 'flatfile_iwate2011_zheng1_srf3d_lasson_0.25Hz_Residuals.csv'
iwa_zh_3d_025Hz_tal = home_3D_025Hz + 'flatfile_iwate2011_zheng1_srf3d_talapas_0.25Hz_Residuals.csv'

miy_usgs_3d_025Hz_las = home_3D_025Hz + 'flatfile_miyagi2011a_usgs_srf3d_lasson_0.25Hz_Residuals.csv'
miy_usgs_3d_025Hz_tal = home_3D_025Hz + 'flatfile_miyagi2011a_usgs_srf3d_talapas_0.25Hz_Residuals.csv'

miy_zh_3d_025Hz_las = home_3D_025Hz + 'flatfile_miyagi2011a_zheng1_srf3d_lasson_0.25Hz_Residuals.csv'
miy_zh_3d_025Hz_tal = home_3D_025Hz + 'flatfile_miyagi2011a_zheng1_srf3d_talapas_0.25Hz_Residuals.csv'

iba_src_3d_025Hz_30km_tal = home_3D_025Hz + 'flatfile_ibaraki2011_srcmod_srf3d_z30km_talapas_0.25Hz_Residuals.csv'
miy_zh_3d_025Hz_30km_tal = home_3D_025Hz + 'flatfile_miyagi2011a_zheng1_srf3d_z30km_talapas_0.25Hz_Residuals.csv'

tok_src3_3d_025Hz_tal = home_3D_025Hz + 'flatfile_tokachi2003_srcmod3_srf3d_talapas_0.25Hz_Residuals.csv'

tok_usgs_3d_025Hz_las = home_3D_025Hz + 'flatfile_tokachi2003_usgs_srf3d_lasson_0.25Hz_Residuals.csv'
tok_usgs_3d_025Hz_tal = home_3D_025Hz + 'flatfile_tokachi2003_usgs_srf3d_talapas_0.25Hz_Residuals.csv'

#%% Creating the Fig 9
#-------Ibaraki 2011-----
fig, axes = plt.subplots(4,4,figsize=(130, 95))
fig.tight_layout(h_pad=60,w_pad=40)#, w_pad=0.5, h_pad=7.0,rect=[0, 0.15, 0.8, 0.9]) #0, 0.03, 0.85, 0.90]

#%% 1D vs 3D (0.25 Hz) Ibaraki 2011 SRCMOD
# # --------Ibaraki 2011 SRCMOD -------------------------------------------------
data =np.array([[iba_src_1d_025Hz,tcb10[0],'MudPy 1D SRCMOD','o'],
                [iba_src_3d_025Hz_tal,tcb10[1],'SW4 3D SCRMOD','x']],dtype=object)

rupt_no = [5,9]
fontsize = 140
funcs.plot_compare_IM_res(axes = axes[0,0],data=data,y_axis="pgd_res",ylim=[-0.5,2],n_rupt=rupt_no,
                        title='Ibaraki 2011 (SRCMOD)',title_pad = 400,subplt_label='A',fontsize=fontsize,
                        tag='2a_pgd_1d_3d_ibaraki_srcmod',xticks=[50,250,450,650,850,1050])              
                                                                                                                                   
funcs.plot_compare_IM_res(axes = axes[1,0],data=data,y_axis="tPGD_res",ylim=[-110,150],n_rupt=rupt_no,
                    xticks=[50,250,450,650,850,1050],subplt_label='E',fontsize=fontsize,
                    tag='2b_tpgd_1d_3d_ibaraki_srcmod',ylabel='Residual (s)')

funcs.plot_compare_IM_res(axes = axes[2,0],data=data,y_axis="sd_res",n_rupt=rupt_no,
                    xticks=[50,250,450,650,850,1050],subplt_label='I',fontsize=fontsize,
                    tag='2c_static_1d_3d_ibaraki_srcmod',ylim=[-3,4])

funcs.plot_compare_IM_res(axes = axes[3,0],data=data,y_axis="xcorr",n_rupt=rupt_no,ylim=[0.6,1],
                    xticks=[50,250,450,650,850,1050],xlabel='distance (km)',fontsize=fontsize,
                    tag='2d_xcorr_1d_3d_ibaraki_srcmod',subplt_label='M')


#%%-------0.25 Hz 1D vs 3D Miyagi 2011 USGS -----------------------------------------------------------------

data =np.array([[miy_usgs_1d_025Hz,tcb10[0],'MudPy 1D HAYES','//'],
                [miy_usgs_3d_025Hz_las,tcb10[1],'SW4 3D HAYES','x']],dtype=object)
rupt_no = [0,1]
xticks = [150,350,550,750,950,1250]
funcs.plot_compare_IM_res(axes = axes[0,1],data=data,y_axis="pgd_res",ylim=[-2,3],n_rupt=rupt_no,fontsize=fontsize,
                    title='Miyagi 2011 (Hayes)',xticks=xticks,title_pad = 400,subplt_label='B')

funcs.plot_compare_IM_res(axes = axes[1,1],data=data,y_axis="tPGD_res",ylim=[-250,100],fontsize=fontsize,
                    n_rupt=rupt_no,ylabel='Residual (s)',xticks=xticks,subplt_label='F')

funcs.plot_compare_IM_res(axes = axes[2,1],data=data,y_axis="sd_res",n_rupt=rupt_no,fontsize=fontsize,
                    ylim=[-5,7],xticks=xticks,subplt_label='J')

funcs.plot_compare_IM_res(axes = axes[3,1],data=data,y_axis="xcorr",n_rupt=rupt_no,ylim=[0.4,1],fontsize=fontsize,
                    xticks=xticks,xlabel='distance (km)',subplt_label='N')


#%%------ 0.25 Hz Iwate 2011 -------------------------------------------------------------------------

data =np.array([[iwa_zh_1d_025Hz,tcb10[0],'MudPy 1D ZHENG','//'],
                [iwa_zh_3d_025Hz_las,tcb10[1],'SW4 3D ZHENG','x']],dtype=object)
rupt_no = [0,1]
xticks = [50,250,450,650,850,1050,1250,1450]
funcs.plot_compare_IM_res(axes = axes[0,2],data=data,y_axis="pgd_res",ylim=[-1,3.5],fontsize=fontsize,
                    title='Iwate 2011 (Zheng)',xticks=xticks,title_pad = 400,subplt_label='C')

funcs.plot_compare_IM_res(axes = axes[1,2],data=data,y_axis="tPGD_res",ylim=[-200,150],fontsize=fontsize,
                    ylabel='Residual (s)',xticks=xticks,subplt_label='G')

funcs.plot_compare_IM_res(axes = axes[2,2],data=data,y_axis="sd_res",fontsize=fontsize,
                    ylim=[-2.5,6],xticks=xticks,subplt_label='K')

funcs.plot_compare_IM_res(axes = axes[3,2],data=data,y_axis="xcorr",ylim=[0.4,1],fontsize=fontsize,
                    xticks=xticks,xlabel='distance (km)',subplt_label='O')

#%% tokachi 2003 usgs talapas
data =np.array([[tok_usgs_1d_025Hz,tcb10[0],'MudPy 1D USGS','//'],
                [tok_usgs_3d_025Hz_tal,tcb10[1],'SW4 3D USGS','x']],dtype=object)
rupt_no = [0,1]
xticks = [50,150,250,350,450,550,650,750]
funcs.plot_compare_IM_res(axes = axes[0,3],data=data,y_axis="pgd_res",ylim=[-2,1.2],n_rupt=rupt_no,fontsize=fontsize,
                    title='Tokachi 2003 (Hayes)',xticks=xticks,title_pad = 400,subplt_label='D')

funcs.plot_compare_IM_res(axes = axes[1,3],data=data,y_axis="tPGD_res",ylim=[-25,75],fontsize=fontsize,
                    n_rupt=rupt_no,ylabel='Residual (s)',xticks=xticks,subplt_label='H')

funcs.plot_compare_IM_res(axes = axes[2,3],data=data,y_axis="sd_res",n_rupt=rupt_no, fontsize=fontsize,
                    ylim=[-6,3],xticks=xticks,subplt_label='L')

funcs.plot_compare_IM_res(axes = axes[3,3],data=data,y_axis="xcorr",n_rupt=rupt_no,ylim=[0.3,1],fontsize=fontsize,
                    xticks=xticks,xlabel='distance (km)',subplt_label='P')

#-------------------------------------------------------------------------
# Creating legend
legend_elements = []
for i in range(2):
    legend_elements.append(Patch(alpha=0.6,edgecolor='black',linewidth=10))
    
hatches = ['o','x']
legend = ['MudPy 1D','SW4 3D']
colors = [tcb10[0],tcb10[1]]

for j, patch in enumerate(legend_elements):
    patch.set_hatch(hatches[j])
    patch.set_label(legend[j])
    patch.set_facecolor(colors[j])
    
legend_elements.append(Line2D([0],[0], linewidth=20.0, linestyle='-',  color='red', alpha=1.0,label='Zero line'))

plt.legend(handles=legend_elements, bbox_to_anchor=(-2.5, -1.05 ), loc='lower left', 
            fontsize=160,frameon=False, ncol=3)

plt.text(-20, -0.15, 'LEGEND', color='k', fontsize=fontsize,fontdict={"weight": "bold"})
plt.text(-14, 4.57, 'PGD Residuals', color='k', fontsize=fontsize+10)
plt.text(-14, 3.47, 'tPGD Residuals', color='k', fontsize=fontsize+10)
plt.text(-14, 2.3, 'SD Residuals', color='k', fontsize=fontsize+10)
plt.text(-14, 1.15, 'Xcorr Residuals', color='k', fontsize=fontsize+10)


plt.show()

figpath = os.getcwd() +'/fig9.1d_vs_3d_residuals.png'

fig.savefig(figpath, bbox_inches='tight', dpi=100)


#%% Creating the Fig S8 
#-------Ibaraki 2011-----
fig, axes = plt.subplots(4,3,figsize=(100, 95))
fig.tight_layout(h_pad=60,w_pad=40)#, w_pad=0.5, h_pad=7.0,rect=[0, 0.15, 0.8, 0.9]) #0, 0.03, 0.85, 0.90]

#%% -------------1D vs 3D (0.25 Hz) Ibaraki 2011 ZHENG
data =np.array([[iba_zh_1d_025Hz,tcb10[0],'MudPy 1D ZHENG','o'],
                [iba_zh_3d_025Hz_tal,tcb10[1],'SW4 3D ZHENG','x']],dtype=object)
rupt_no = [0,1]
xticks=[50,250,450,650,850,1050]
funcs.plot_compare_IM_res(axes = axes[0,0],data=data,y_axis="pgd_res",ylim=[-1.5,2],n_rupt=rupt_no,
                        title='Ibaraki 2011 (Zheng)',xticks=xticks,title_pad = 400,subplt_label='A')
                                                                                                                                                       
funcs.plot_compare_IM_res(axes = axes[1,0],data=data,y_axis="tPGD_res",ylim=[-100,120],
                    n_rupt=rupt_no,ylabel='Residual (s)',subplt_label='D',xticks=xticks)

funcs.plot_compare_IM_res(axes = axes[2,0],data=data,y_axis="sd_res",n_rupt=rupt_no,
                    ylim=[-2,5],subplt_label='G',xticks=xticks)

funcs.plot_compare_IM_res(axes = axes[3,0],data=data,y_axis="xcorr",n_rupt=rupt_no,
                          ylim=[0.5,1],subplt_label='J',xticks=xticks,xlabel='distance (km)')


#%%-------0.25 Hz 1D vs 3D Miyagi 2011 Zheng------------------------------------------------------------------------
data =np.array([[miy_zh_1d_025Hz,tcb10[0],'MudPy 1D ZHENG','//'],
                [miy_zh_3d_025Hz_las,tcb10[1],'SW4 3D ZHENG','x']],dtype=object)
rupt_no = [0,1]
xticks = [150,350,550,750,950,1250]
funcs.plot_compare_IM_res(axes = axes[0,1],data=data,y_axis="pgd_res",ylim=[-2,3],
                    n_rupt=rupt_no,title='Miyagi 2011 (Zheng)',xticks=xticks,title_pad = 400,subplt_label='B')

funcs.plot_compare_IM_res(axes = axes[1,1],data=data,y_axis="tPGD_res",ylim=[-300,100],
                    n_rupt=rupt_no, ylabel='Residual (s)',xticks=xticks,subplt_label='E')

funcs.plot_compare_IM_res(axes = axes[2,1],data=data,y_axis="sd_res",
                          n_rupt=rupt_no,ylim=[-7,7],xticks=xticks,subplt_label='H')

funcs.plot_compare_IM_res(axes = axes[3,1],data=data,y_axis="xcorr",subplt_label='K',
                          n_rupt=rupt_no,ylim=[0.4,1],xticks=xticks,xlabel='distance (km)')


#%%--------0.25Hz 1d vs 3d Tokachi 2003 SRCMOD 3talapas -----------------------------------------------------------------------

data =np.array([[tok_src3_1d_025Hz,tcb10[0],'MudPy 1D SRCMOD3','//'],
                [tok_src3_3d_025Hz_tal,tcb10[1],'SW4 3D SRCMOD3','x']],dtype=object)
rupt_no = [1,4]
xticks = [50,150,250,350,450,550,650,750]
funcs.plot_compare_IM_res(axes = axes[0,2],data=data,y_axis="pgd_res",ylim=[-1.5,1.2],n_rupt=rupt_no,
                    title='Tokachi 2003 (SRCMOD3)',xticks=xticks,title_pad = 400,subplt_label='C')

funcs.plot_compare_IM_res(axes = axes[1,2],data=data,y_axis="tPGD_res",ylim=[-10,75],
                    n_rupt=rupt_no,ylabel='Residual (s)',xticks=xticks,subplt_label='F')

funcs.plot_compare_IM_res(axes = axes[2,2],data=data,y_axis="sd_res",n_rupt=rupt_no,
                          ylim=[-6,4],xticks=xticks,subplt_label='I')

funcs.plot_compare_IM_res(axes = axes[3,2],data=data,y_axis="xcorr",n_rupt=rupt_no,
                          ylim=[0.3,1],xticks=xticks,xlabel='distance (km)',subplt_label='L')


#-------------------------------------------------------------------------
# Creating legend
legend_elements = []
for i in range(2):
    legend_elements.append(Patch(alpha=0.6,edgecolor='black',linewidth=10))
    
hatches = ['o','x']
legend = ['MudPy 1D','SW4 3D']
colors = [tcb10[0],tcb10[1]]

for j, patch in enumerate(legend_elements):
    patch.set_hatch(hatches[j])
    patch.set_label(legend[j])
    patch.set_facecolor(colors[j])
    
legend_elements.append(Line2D([0],[0], linewidth=20.0, linestyle='-',  color='red', alpha=1.0,label='Zero line'))

plt.legend(handles=legend_elements, bbox_to_anchor=(-2.0, -1.05 ), loc='lower left', 
            fontsize=160,frameon=False, ncol=3)

plt.text(-16, -0.15, 'LEGEND', color='k', fontsize=130,fontdict={"weight": "bold"})
plt.text(-9, 4.57, 'PGD Residuals', color='k', fontsize=160)
plt.text(-9, 3.47, 'tPGD Residuals', color='k', fontsize=160)
plt.text(-9, 2.3, 'SD Residuals', color='k', fontsize=160)
plt.text(-9, 1.15, 'Xcorr Residuals', color='k', fontsize=160)


plt.show()

figpath = os.getcwd() +'/figS8.1d_vs_3d_residuals.png'

fig.savefig(figpath, bbox_inches='tight', dpi=100)