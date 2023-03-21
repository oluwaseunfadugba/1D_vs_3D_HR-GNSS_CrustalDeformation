#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 16:47:04 2023

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


iba_src_1d_050Hz = home_1D+'flatfile_1d_ibaraki2011_srcmod_0.49Hz_Residuals.csv'

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


#%% 3D 0.50Hz
home_3D_050Hz = "/Users/oluwaseunfadugba/Documents/Projects/TsE_ValerieDiego/TsE_1D_vs_3D/3D_Modeling_using_SW4/Running_Simulations/4_Model_results/All_05Hz_Results/"


iba_src_3d_050Hz = home_3D_050Hz + 'flatfile_ibaraki2011_srcmod_srf3d_talapas_0.49Hz_Residuals.csv'
iba_zh_3d_050Hz = home_3D_050Hz + 'flatfile_ibaraki2011_zheng1_srf3d_talapas_0.49Hz_Residuals.csv'

iwa_zh_3d_050Hz = home_3D_050Hz + 'flatfile_iwate2011_zheng1_srf3d_talapas_0.49Hz_Residuals.csv'

miy_usgs_3d_050Hz = home_3D_050Hz + 'flatfile_miyagi2011a_usgs_srf3d_talapas_0.49Hz_Residuals.csv'
miy_zh_3d_050Hz = home_3D_050Hz + 'flatfile_miyagi2011a_zheng1_srf3d_talapas_0.49Hz_Residuals.csv'

tok_src3_3d_050Hz = home_3D_050Hz + 'flatfile_tokachi2003_srcmod3_srf3d_talapas_0.49Hz_Residuals.csv'
tok_usgs_3d_050Hz = home_3D_050Hz + 'flatfile_tokachi2003_usgs_srf3d_talapas_0.49Hz_Residuals.csv'





#%% Creating the Fig 13
#-------Ibaraki 2011-----
fig, axes = plt.subplots(2,3,figsize=(100, 60))
fig.tight_layout(h_pad=60,w_pad=40)#, w_pad=0.5, h_pad=7.0,rect=[0, 0.15, 0.8, 0.9]) #0, 0.03, 0.85, 0.90]


#%% 1D vs 3D (0.25 vs 0.5 Hz) Ibaraki 2011 SRCMOD
# # --------Ibaraki 2011-----------------------------------------------------------------------
data =np.array([[iba_src_1d_025Hz,tcb10[0],'MudPy 1D SRCMOD','o'],
                [iba_src_3d_025Hz_tal,tcb10[1],'SW4 3D SCRMOD (0.25 Hz)','x'],
                [iba_src_3d_050Hz,tcb10[2],'SW4 3D SCRMOD (0.5 Hz)','+']],dtype=object)

xticks=[50,250,450,650,850,1050]

funcs.plot_compare_IM_res(axes = axes[0,0],data=data,y_axis="pgd_res",ylim=[-0.5,2],
                          n_rupt=[5],title='PGD Residual',xticks=xticks,
                          subplt_label='A',title_pad = 70)

funcs.plot_compare_IM_res(axes = axes[0,1],data=data,y_axis="tPGD_res",ylim=[-110,125],
                          n_rupt=[5],title='tPGD Residual',ylabel='Residual (s)',
                          xticks=xticks,subplt_label='B',title_pad = 70)

funcs.plot_compare_IM_res(axes = axes[1,0],data=data,y_axis="sd_res",n_rupt=[5],
                          subplt_label='C',title='Static disp Residual',ylim=[-2,4],
                          xticks=xticks,xlabel='distance (km)',title_pad = 70)

funcs.plot_compare_IM_res(axes = axes[1,1],data=data,y_axis="xcorr",n_rupt=[5],
                          ylim=[0.6,1],title='Cross Correlation',xlabel='distance (km)',
                          xticks=xticks,subplt_label='D',title_pad = 70,ylabel='value')

models = np.array([[iba_src_1d_025Hz, iba_src_3d_025Hz_tal, 'SW4_025_Hz',[0], [5]],
                   [iba_src_1d_050Hz, iba_src_3d_050Hz, 'SW4_050_Hz',[0], [5]]],dtype=object)

kstest_ff_path='/Users/oluwaseunfadugba/Documents/Projects/TsE_ValerieDiego/TsE_1D_vs_3D/'+\
    'Paper_Figures/Comparing_kstest_Residuals_with_median_diff_0_25vs05Hz/'
    
funcs.plot_median_residual(axes=axes[1,2],models=models,attribute='pgd_res',title_pad = 70,
                           xlabel='distance (km)',xticks=xticks,yticks=[0,-0.5,-1],subplt_label='E',
                           title='PGD 3D-1D Res. Median',kstest_ff_path=kstest_ff_path)

axes[0,2].axis('off')

#-------------------------------------------------------------------------
# Creating legend
legend_elements = []
for i in range(3):
    legend_elements.append(Patch(alpha=0.6,edgecolor='black',linewidth=10))
    
hatches = ['o','x','+']
legend = ['MudPy 1D SRCMOD','SW4 3D SRCMOD (0.25 Hz)','SW4 3D SRCMOD (0.5 Hz)']
colors = [tcb10[0],tcb10[1],tcb10[2]]

for j, patch in enumerate(legend_elements):
    patch.set_hatch(hatches[j])
    patch.set_label(legend[j])
    patch.set_facecolor(colors[j])
    
legend_elements.append(Line2D([0],[0], linewidth=20.0, linestyle='-',  color='red', 
                              alpha=1.0,label='Zero line'))

legend_elements.append(Line2D([0],[0], linewidth=20.0, linestyle='-',  color='steelblue', 
                              alpha=0,label='$\delta_{|3D|-|1D|}$'))

legend_elements.append(Line2D([0],[0], linewidth=20.0, linestyle='-',  color='steelblue', 
                              alpha=1.0,label='fmax = 0.25 Hz'))

legend_elements.append(Line2D([0],[0], linewidth=20.0, linestyle='--',  color='darkorange', 
                              alpha=1.0,label='fmax = 0.5 Hz'))

plt.legend(handles=legend_elements, bbox_to_anchor=(-0.15, 2.35 ), loc='upper left', 
            fontsize=160,frameon=False)

plt.text(-6, 1.8, 'LEGEND', color='k', fontsize=130,fontdict={"weight": "bold"})



plt.show()

figpath = os.getcwd() +'/fig13.0.25_vs_0.5Hz_residuals.png'

fig.savefig(figpath, bbox_inches='tight', dpi=100)



#%% Creating the Fig 14

fig, axes = plt.subplots(3,3,figsize=(90, 80))
fig.tight_layout(h_pad=60,w_pad=40)#, w_pad=0.5, h_pad=7.0,rect=[0, 0.15, 0.8, 0.9]) #0, 0.03, 0.85, 0.90]

#% 1D vs 3D (0.25 vs 0.5 Hz) Ibaraki 2011 SRCMOD
data =np.array([[iba_src_1d_025Hz,tcb10[0],'MudPy 1D SRCMOD','o'],
                [iba_src_3d_025Hz_tal,tcb10[1],'SW4 3D SCRMOD (0.25 Hz)','x'],
                [iba_src_3d_050Hz,tcb10[2],'SW4 3D SCRMOD (0.5 Hz)','+']],dtype=object)

funcs.plot_compare_IM_res(axes = axes[0,0],data=data,y_axis="pgd_res",ylim=[-0.5,2],
                          n_rupt=[5],title='Ibaraki 2011 SRCMOD',xticks=[50,250,450,650,850,1050],
                          subplt_label='A',title_pad = 70)


#% 1D vs 3D (0.25 vs 0.5 Hz) Ibaraki 2011 ZHENG
data =np.array([[iba_zh_1d_025Hz,tcb10[0],'MudPy 1D ZHENG','o'],
                [iba_zh_3d_025Hz_tal,tcb10[1],'SW4 3D ZHENG (0.25 Hz)','x'],
                [iba_zh_3d_050Hz,tcb10[2],'SW4 3D ZHENG (0.50 Hz)','+']],dtype=object)

funcs.plot_compare_IM_res(axes = axes[0,1],data=data,y_axis="pgd_res",ylim=[-1.5,2],n_rupt=[1],
                        title='Ibaraki 2011 Zheng',xticks=[50,250,450,650,850,1050],
                        subplt_label='B',title_pad = 70)

#%- 1D vs 3D (0.25 vs 0.5 Hz) Iwate 2011 Zheng

data =np.array([[iwa_zh_1d_025Hz,tcb10[0],'MudPy 1D ZHENG','o'],
                [iwa_zh_3d_025Hz_las,tcb10[1],'SW4 3D ZHENG (0.25 Hz)','x'],
                [iwa_zh_3d_050Hz,tcb10[2],'SW4 3D ZHENG (0.50 Hz)','+']],dtype=object)

funcs.plot_compare_IM_res(axes = axes[0,2],data=data,y_axis="pgd_res",ylim=[-1,2.5],n_rupt=[0],
                    title='Iwate 2011 Zheng',xticks=[50,250,450,650,850,1050],
                    subplt_label='C',title_pad = 70,xlabel='distance (km)')

#%-------1D vs 3D (0.25 vs 0.5 Hz) Miyagi 2011 Zheng -----------------------------------------------------------------
data =np.array([[miy_zh_1d_025Hz,tcb10[0],'MudPy 1D ZHENG','o'],
                [miy_zh_3d_025Hz_las,tcb10[1],'SW4 3D ZHENG (0.25 Hz)','x'],
                [miy_zh_3d_050Hz,tcb10[2],'SW4 3D ZHENG (0.50 Hz)','+']],dtype=object)

funcs.plot_compare_IM_res(axes = axes[1,0],data=data,y_axis="pgd_res",ylim=[-1,3],n_rupt=[0],
                    title='Miyagi 2011 ZHeng',xticks=[150,350,550,750,950],
                    subplt_label='D',title_pad = 70)

#%-------1D vs 3D (0.25 vs 0.5 Hz) Miyagi 2011 USGS -----------------------------------------------------------------

data =np.array([[miy_usgs_1d_025Hz,tcb10[0],'MudPy 1D HAYES','o'],
                [miy_usgs_3d_025Hz_las,tcb10[1],'SW4 3D HAYES (0.25 Hz)','x'],
                [miy_usgs_3d_050Hz,tcb10[2],'SW4 3D HAYES (0.50 Hz)','+'],],dtype=object)

funcs.plot_compare_IM_res(axes = axes[1,1],data=data,y_axis="pgd_res",ylim=[-1.5,2.5],n_rupt=[1],
                    title='Miyagi 2011 Hayes',xticks=[150,350,550,750,950],
                    subplt_label='E',title_pad = 70)


#%% 1d vs 3d (0.25 vs 0.5 Hz) tokachi 2003 usgs talapas
data =np.array([[tok_usgs_1d_025Hz,tcb10[0],'MudPy 1D USGS','o'],
                [tok_usgs_3d_025Hz_tal,tcb10[1],'SW4 3D USGS (0.25 Hz)','x'],
                [tok_usgs_3d_050Hz,tcb10[2],'SW4 3D USGS (0.50 Hz)','+']],dtype=object)

funcs.plot_compare_IM_res(axes = axes[2,0],data=data,y_axis="pgd_res",ylim=[-4,1.2],n_rupt=[0],
                    subplt_label='F',title='Tokachi 2003 Hayes',title_pad = 70,xlabel='distance (km)',xticks=[50,150,250,350,450,550,650,750])


#%%-------- 1d vs 3d (0.25 vs 0.5 Hz) Tokachi 2003 SRCMOD 3talapas -----------------------------------------------------------------------

data =np.array([[tok_src3_1d_025Hz,tcb10[0],'MudPy 1D SRCMOD3','o'],
                [tok_src3_3d_025Hz_tal,tcb10[1],'SW4 3D SRCMOD3 (0.25 Hz)','x'],
                [tok_src3_3d_050Hz,tcb10[2],'SW4 3D SRCMOD3 (0.50 Hz)','+']],dtype=object)

funcs.plot_compare_IM_res(axes = axes[2,1],data=data,y_axis="pgd_res",ylim=[-1.5,1.5],n_rupt=[1],
                    subplt_label='G',title='Tokachi 2003 SRCMOD3',title_pad = 70,xticks=[50,150,250,350,450,550,650,750],xlabel='distance (km)')
                    

axes[1,2].axis('off')
axes[2,2].axis('off')

#-------------------------------------------------------------------------
# Creating legend
legend_elements = []
for i in range(3):
    legend_elements.append(Patch(alpha=0.6,edgecolor='black',linewidth=10))
    
hatches = ['o','x','+']
legend = ['MudPy 1D SRCMOD','SW4 3D SRCMOD (0.25 Hz)','SW4 3D SRCMOD (0.5 Hz)']
colors = [tcb10[0],tcb10[1],tcb10[2]]

for j, patch in enumerate(legend_elements):
    patch.set_hatch(hatches[j])
    patch.set_label(legend[j])
    patch.set_facecolor(colors[j])
    
legend_elements.append(Line2D([0],[0], linewidth=20.0, linestyle='-',  color='red', 
                              alpha=1.0,label='Zero line'))

plt.legend(handles=legend_elements, bbox_to_anchor=(-0.2, 2.2 ), loc='upper left', 
            fontsize=160,frameon=False)

plt.text(-0.15, 2.45, 'LEGEND', color='k', fontsize=130,fontdict={"weight": "bold"})
plt.text(-0.15, 2.25, 'PGD Residual only', color='k', fontsize=130,fontdict={"weight": "bold"})



plt.show()

figpath = os.getcwd() +'/fig14.0.25_vs_0.5Hz_PGD_residuals.png'

fig.savefig(figpath, bbox_inches='tight', dpi=100)

