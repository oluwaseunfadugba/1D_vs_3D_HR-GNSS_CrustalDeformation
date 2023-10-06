#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 14:37:16 2023

@author: oluwaseunfadugba
"""

import numpy as np
import time
import matplotlib.pyplot as plt
# import seaborn as sns
# import matplotlib.pyplot as plt
# import pandas as pd
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

iba_src_vjma1d_025Hz = home_1D+'flatfile_1d_ibaraki2011_srcmod_vjma_vel_0.25Hz_Residuals.csv'
iba_src_zheng1d_025Hz = home_1D+'flatfile_1d_ibaraki2011_srcmod_zheng_vel_0.25Hz_Residuals.csv'
iba_src_kok1d_025Hz = home_1D+'flatfile_1d_ibaraki2011_srcmod_kok_vel_0.25Hz_Residuals.csv'
iba_src_hayes1d_025Hz = home_1D+'flatfile_1d_ibaraki2011_srcmod_0.25Hz_Residuals.csv'


#%% Creating the subplots (Four 1D velocity Models)
#-------Ibaraki 2011-----
fig, axes = plt.subplots(1,1,figsize=(70, 50))
fig.tight_layout(h_pad=40,w_pad=40)#, w_pad=0.5, h_pad=7.0,rect=[0, 0.15, 0.8, 0.9]) #0, 0.03, 0.85, 0.90]

data =np.array([[iba_src_vjma1d_025Hz,tcb10[5],'Ueno et al. (2002)','+'],
                [iba_src_zheng1d_025Hz,tcb10[7],'Zheng et al. (2020)','o'],
                [iba_src_kok1d_025Hz,tcb10[2],   'Koketsu et al. (2004)',   '/'],
                [iba_src_hayes1d_025Hz,tcb10[0],'Hayes (2017)','o']],dtype=object)
funcs.plot_compare_IM_res(axes = axes,data=data,y_axis="pgd_res",ylim=[-1.0,3],
                    title='Effect of 1D Model using Ibaraki 2011 (SRCMOD)',xticks=[50,250,450,650,850,1050,1350],
                    tag='1d_ibaraki',title_pad = 200,fontsize=220)

#-------------------------------------------------------------------------
# Creating legend
legend_elements = []
for i in range(len(data)):
    legend_elements.append(Patch(alpha=0.6,edgecolor='black',linewidth=10))
    
hatches = ['o','/','o','+']
legend = ['Hayes (2017)','Koketsu et al. (2004)','Zheng et al. (2020)','Ueno et al. (2002)']
colors = [tcb10[0],tcb10[2],tcb10[7],tcb10[5]]

for j, patch in enumerate(legend_elements):
    patch.set_hatch(hatches[j])
    patch.set_label(legend[j])
    patch.set_facecolor(colors[j])
    
legend_elements.append(Line2D([0],[0], linewidth=20.0, linestyle='-',  color='red', alpha=1.0,label='Zero line'))

plt.legend(handles=legend_elements, bbox_to_anchor=(1.05, 0.65), loc='upper left', 
            fontsize=170,frameon=False)

plt.text(13.5, 2.0, 'LEGEND', color='k', fontsize=200,fontdict={"weight": "bold"})
plt.text(13.5, 1.7, 'PGD Residuals (1D only)', color='k', fontsize=200)


plt.show()
figpath = os.getcwd() +'/fig6_supl.1d_pgd_residuals.png'

fig.savefig(figpath, bbox_inches='tight', dpi=100)


#%% Creating the subplots (Three 1D velocity Models)
#-------Ibaraki 2011-----
fig, axes = plt.subplots(1,1,figsize=(70, 50))
fig.tight_layout(h_pad=40,w_pad=40)#, w_pad=0.5, h_pad=7.0,rect=[0, 0.15, 0.8, 0.9]) #0, 0.03, 0.85, 0.90]

data =np.array([#[iba_src_vjma1d_025Hz,tcb10[5],'Ueno et al. (2002)','+'],
                [iba_src_zheng1d_025Hz,tcb10[7],'Zheng et al. (2020)','o'],
                [iba_src_kok1d_025Hz,tcb10[5],   'Koketsu et al. (2004)',   '/'],
                [iba_src_hayes1d_025Hz,tcb10[0],'Hayes (2017)','o']],dtype=object)
funcs.plot_compare_IM_res(axes = axes,data=data,y_axis="pgd_res",ylim=[-1.0,3],
                    title='Effect of 1D Model using Ibaraki 2011 (SRCMOD)',xticks=[50,250,450,650,850,1050,1350],
                    tag='1d_ibaraki',title_pad = 200,fontsize=220)

#-------------------------------------------------------------------------
# Creating legend
legend_elements = []
for i in range(len(data)):
    legend_elements.append(Patch(alpha=0.6,edgecolor='black',linewidth=10))
    
hatches = ['o','/','o','+']
legend = ['Hayes (2017)','Koketsu et al. (2004)','Zheng et al. (2020)']
colors = [tcb10[0],tcb10[5],tcb10[7]]

for j, patch in enumerate(legend_elements):
    patch.set_hatch(hatches[j])
    patch.set_label(legend[j])
    patch.set_facecolor(colors[j])
    
legend_elements.append(Line2D([0],[0], linewidth=20.0, linestyle='-',  color='red', alpha=1.0,label='Zero line'))

plt.legend(handles=legend_elements, bbox_to_anchor=(1.05, 0.65), loc='upper left', 
            fontsize=170,frameon=False)

plt.text(13.5, 2.0, 'LEGEND', color='k', fontsize=200,fontdict={"weight": "bold"})
plt.text(13.5, 1.7, 'PGD Residuals (1D only)', color='k', fontsize=200)


plt.show()
figpath = os.getcwd() +'/fig6_supl_2.1d_pgd_residuals.png'

fig.savefig(figpath, bbox_inches='tight', dpi=100)
