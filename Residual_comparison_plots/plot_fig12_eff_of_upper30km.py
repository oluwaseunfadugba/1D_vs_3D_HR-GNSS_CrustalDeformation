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
from matplotlib.cbook import get_sample_data

start = time.time()

import os
import comparing_funcs as funcs


tcb10 = ['#006BA4', '#FF800E', '#ABABAB', '#595959', '#5F9ED1', '#C85200', '#898989', '#A2C8EC', '#FFBC79', '#CFCFCF']
plt.style.use('tableau-colorblind10') #('seaborn-colorblind') #

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


iba_src_1d_050Hz = home_1D+'flatfile_1d_ibaraki2011_srcmod_0.49Hz_Residuals.csv'


#%% 3D 0.50Hz
home_3D_050Hz = "/Users/oluwaseunfadugba/Documents/Projects/TsE_ValerieDiego/TsE_1D_vs_3D/3D_Modeling_using_SW4/Running_Simulations/4_Model_results/All_05Hz_Results/"


iba_src_3d_050Hz = home_3D_050Hz + 'flatfile_ibaraki2011_srcmod_srf3d_talapas_0.49Hz_Residuals.csv'
iba_zh_3d_050Hz = home_3D_050Hz + 'flatfile_ibaraki2011_zheng1_srf3d_talapas_0.49Hz_Residuals.csv'

iwa_zh_3d_050Hz = home_3D_050Hz + 'flatfile_iwate2011_zheng1_srf3d_talapas_0.49Hz_Residuals.csv'

miy_usgs_3d_050Hz = home_3D_050Hz + 'flatfile_miyagi2011a_usgs_srf3d_talapas_0.49Hz_Residuals.csv'
miy_zh_3d_050Hz = home_3D_050Hz + 'flatfile_miyagi2011a_zheng1_srf3d_talapas_0.49Hz_Residuals.csv'

tok_src3_3d_050Hz = home_3D_050Hz + 'flatfile_tokachi2003_srcmod3_srf3d_talapas_0.49Hz_Residuals.csv'
tok_usgs_3d_050Hz = home_3D_050Hz + 'flatfile_tokachi2003_usgs_srf3d_talapas_0.49Hz_Residuals.csv'



#%% Creating the Fig S8 
#-------Ibaraki 2011-----
fig, axes = plt.subplots(4,3,figsize=(100, 95))
fig.tight_layout(h_pad=60,w_pad=40)#, w_pad=0.5, h_pad=7.0,rect=[0, 0.15, 0.8, 0.9]) #0, 0.03, 0.85, 0.90]

#%% Effect of the upper 30km 3D structure
data =np.array([[iba_src_1d_025Hz,tcb10[0],'MudPy 1D SRCMOD','/'],
                [iba_src_3d_025Hz_tal,tcb10[1],'SW4 3D SRCMOD','x'],
                [iba_src_3d_025Hz_30km_tal,tcb10[3],'SW4 3D SRCMOD (3D to 30 km)','o']],dtype=object)

rupt_no = [5,9]
xticks = [50,250,450,650,850,1050]
funcs.plot_compare_IM_res(axes = axes[0,0],data=data,y_axis="pgd_res",ylim=[-0.5,2.1],n_rupt=rupt_no,
                    title='Ibaraki 2011 SRCMOD',title_pad = 550,xticks=xticks,subplt_label='A')

funcs.plot_compare_IM_res(axes = axes[1,0],data=data,y_axis="tPGD_res",ylim=[-150,150],
                    n_rupt=rupt_no,ylabel='Residual (s)',xticks=xticks,subplt_label='D')

funcs.plot_compare_IM_res(axes = axes[2,0],data=data,y_axis="sd_res",n_rupt=rupt_no,
                    ylim=[-3,4],xticks=xticks,subplt_label='G')

funcs.plot_compare_IM_res(axes = axes[3,0],data=data,y_axis="xcorr",n_rupt=rupt_no,
                          ylim=[0.6,1],xticks=xticks,ylabel='value',xlabel='distance (km)',subplt_label='J')


#%% Effect of the upper 30km 3D structure
data =np.array([[miy_zh_1d_025Hz,tcb10[0],'MudPy 1D ZHENG','/'],
                [miy_zh_3d_025Hz_las,tcb10[1],'SW4 3D ZHENG','x'],
                [miy_zh_3d_025Hz_30km_tal,tcb10[3],'SW4 3D ZHENG (3D to 30 km)','o']],dtype=object)

rupt_no = [0,3]
xticks = [150,350,550,750,950]
funcs.plot_compare_IM_res(axes = axes[0,1],data=data,y_axis="pgd_res",ylim=[-1,3],n_rupt=rupt_no,
                    title='Miyagi 2011 ZHENG',xticks=xticks,subplt_label='B',title_pad = 550)

funcs.plot_compare_IM_res(axes = axes[1,1],data=data,y_axis="tPGD_res",ylim=[-250,100],
                    n_rupt=rupt_no,ylabel='Residual (s)',xticks=xticks,subplt_label='E')

funcs.plot_compare_IM_res(axes = axes[2,1],data=data,y_axis="sd_res",n_rupt=rupt_no,
                    ylim=[-4,7],xticks=xticks,subplt_label='H')

funcs.plot_compare_IM_res(axes = axes[3,1],data=data,y_axis="xcorr",n_rupt=rupt_no,
                          ylim=[0.4,1],xticks=xticks,xlabel='distance (km)',subplt_label='K',ylabel='value')

#%% Effect of the upper 30km 3D structure - KSTEST RESULTS

models = np.array([[iba_src_1d_025Hz, iba_src_3d_025Hz_tal, 'ibaraki_srcmod_tal',        [0], [5,9]],
                   [iba_src_1d_025Hz, iba_src_3d_025Hz_30km_tal, 'ibaraki_srcmod_tal_z30km',        [0], [5,9]],
                   [miy_zh_1d_025Hz, miy_zh_3d_025Hz_30km_tal,   'miyagi_zheng_tal_z30km',[0], [0,3]],
                   [miy_zh_1d_025Hz, miy_zh_3d_025Hz_tal, 'miyagi_zheng_tal',        [0], [0,3]]],dtype=object)

kstest_ff_path_z30='/Users/oluwaseunfadugba/Documents/Projects/TsE_ValerieDiego/'+\
    'TsE_1D_vs_3D/Paper_Figures/Comparing_kstest_Residuals_with_median_diff_z30km/'
    
xticks = [50,250,450,650,850,1050]
funcs.plot_median_residual(axes=axes[0,2],models=models,attribute='pgd_res',title_pad = 550,
                           xticks=xticks,yticks=[0,-0.5,-1],subplt_label='C',
                           title='Median Residual Difference',kstest_ff_path=kstest_ff_path_z30)

funcs.plot_median_residual(axes=axes[1,2],models=models,attribute='tPGD_res',
                           xticks=xticks,yticks=[50,0,-50,-100],subplt_label='F',
                           kstest_ff_path=kstest_ff_path_z30,ylabel='s')

funcs.plot_median_residual(axes=axes[2,2],models=models,attribute='sd_res',
                           xticks=xticks,yticks=[1,0,-1,-2,-3],subplt_label='I',
                           kstest_ff_path=kstest_ff_path_z30)

funcs.plot_median_residual(axes=axes[3,2],models=models,attribute='xcorr',rec_neg=0,
                           xticks=xticks,yticks=[0.1,0,-0.1],subplt_label='L',ylabel='value',
                           kstest_ff_path=kstest_ff_path_z30,xlabel='distance (km)')




# #-------------------------------------------------------------------------
# Creating legend
legend_elements = []
for i in range(3):
    legend_elements.append(Patch(alpha=0.6,edgecolor='black',linewidth=10))
    
hatches = ['/','x','o']
legend = ['MudPy 1D','SW4 3D (3D to 200 km depth)','SW4 3D (3D to 30 km depth)']
colors = [tcb10[0],tcb10[1],tcb10[3]]

for j, patch in enumerate(legend_elements):
    patch.set_hatch(hatches[j])
    patch.set_label(legend[j])
    patch.set_facecolor(colors[j])
    
legend_elements.append(Line2D([0],[0], linewidth=20.0, linestyle='-',  color='red', alpha=1.0,label='Zero line'))

legend_elements.append(Line2D([0],[0], linewidth=20.0, linestyle='-',  color='steelblue', 
                              alpha=0,label='$\delta_{|3D|-|1D|}$'))

legend_elements.append(Line2D([0],[0], linewidth=20.0, linestyle='-',  color='steelblue', 
                              alpha=1.0,label='Ibaraki SRCMOD (3D to 200 km)'))

legend_elements.append(Line2D([0],[0], linewidth=20.0, linestyle='--',  color='darkorange', 
                              alpha=1.0,label='Ibaraki SRCMOD (3D to 30 km)'))

legend_elements.append(Line2D([0],[0], linewidth=20.0, linestyle='-',  color='gray', 
                              alpha=1.0,label='Miyagi Zheng (3D to 200 km)'))

legend_elements.append(Line2D([0],[0], linewidth=20.0, linestyle='--',  color='darkgray', 
                              alpha=1.0,label='Miyagi Zheng (3D to 30 km)'))

legend_elements.append(Line2D([0],[0], linewidth=20.0, linestyle='--',  color='black', 
                              alpha=1.0,label='$\delta_{|3D|-|1D|}$ = 0'))

plt.legend(handles=legend_elements, bbox_to_anchor=(1.1, 5.5 ), loc='upper left', 
            fontsize=160,frameon=False,labelspacing = 1)

plt.text(1250.0, 1.03, 'LEGEND', color='k', fontsize=130,fontdict={"weight": "bold"})
plt.text(-1100, 1.13, 'PGD Residuals', color='k', fontsize=160)
plt.text(-1100, 0.8, 'tPGD Residuals', color='k', fontsize=160)
plt.text(-1100, 0.47, 'SD Residuals', color='k', fontsize=160)
plt.text(-1100, .14, 'Xcorr Residuals', color='k', fontsize=160)

# Adding mean difference residual cartoon figure
im = plt.imread(get_sample_data(os.getcwd()+'/mean_diff_cartoon.png'))
newax = fig.add_axes([1.01, 0, 0.4, 0.4], anchor='NE', zorder=-1)
newax.imshow(im)
newax.axis('off')


plt.show()

figpath = os.getcwd() +'/fig12.eff_of_upper30km.png'

fig.savefig(figpath, bbox_inches='tight', dpi=100)