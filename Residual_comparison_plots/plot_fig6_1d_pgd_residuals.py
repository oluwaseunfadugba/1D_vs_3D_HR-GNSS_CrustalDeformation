#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 17:34:31 2023

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

iba_src_1d_025Hz = home_1D+'flatfile_1d_ibaraki2011_srcmod_0.25Hz_Residuals.csv'
iba_zh_1d_025Hz = home_1D+'flatfile_1d_ibaraki2011_zheng1_0.25Hz_Residuals.csv'

iwa_zh_1d_025Hz = home_1D+'flatfile_1d_iwate2011_zheng1_0.25Hz_Residuals.csv'

miy_usgs_1d_025Hz = home_1D+'flatfile_1d_miyagi2011a_usgs_0.25Hz_Residuals.csv'
miy_zh_1d_025Hz = home_1D+'flatfile_1d_miyagi2011a_zheng1_0.25Hz_Residuals.csv'

tok_src1_1d_025Hz = home_1D+'flatfile_1d_tokachi2003_srcmod1_0.25Hz_Residuals.csv'
tok_src2_1d_025Hz = home_1D+'flatfile_1d_tokachi2003_srcmod2_0.25Hz_Residuals.csv'
tok_src3_1d_025Hz = home_1D+'flatfile_1d_tokachi2003_srcmod3_0.25Hz_Residuals.csv'
tok_usgs_1d_025Hz = home_1D+'flatfile_1d_tokachi2003_usgs_0.25Hz_Residuals.csv'

# Creating the subplots
#-------Ibaraki 2011-----
fig, axes = plt.subplots(2,2,figsize=(70, 50))
fig.tight_layout(h_pad=40,w_pad=40)#, w_pad=0.5, h_pad=7.0,rect=[0, 0.15, 0.8, 0.9]) #0, 0.03, 0.85, 0.90]

data =np.array([[iba_src_1d_025Hz,tcb10[0],'MudPy 1D SRCMOD','o'],
                [iba_zh_1d_025Hz,tcb10[1],'MudPy 1D ZHENG','x']],dtype=object)
funcs.plot_compare_IM_res(axes = axes[0,0],data=data,y_axis="pgd_res",ylim=[-1.5,3],
                    title='Ibaraki 2011',xticks=[50,250,450,650,850,1050,1350],
                    tag='1d_ibaraki')

#------Miyagi 2011---------
data =np.array([[miy_usgs_1d_025Hz,tcb10[2],'MudPy 1D USGS','/'],
                [miy_zh_1d_025Hz,tcb10[1],'MudPy 1D ZHENG','x']],dtype=object)
funcs.plot_compare_IM_res(axes = axes[0,1],data=data,y_axis="pgd_res",ylim=[-1,3],
                    title='Miyagi 2011',xticks=[150,350,550,750,950,1250],
                    tag='1d_miyagi')

#------Iwate 2011-----------
data =np.array([[iwa_zh_1d_025Hz,tcb10[1],'MudPy 1D ZHENG','x']],dtype=object)
funcs.plot_compare_IM_res(axes = axes[1,0],data=data,y_axis="pgd_res",ylim=[-0.5,3.1],
                    title='Iwate 2011',xlabel='distance (km)',
                    tag='1d_iwate',xticks=[50,250,450,650,850,1050,1250,1450,1750])

#-------Tokachi 2003----------
data =np.array([[tok_src2_1d_025Hz,tcb10[5],'MudPy 1D SRCMOD2','+'],
                [tok_src3_1d_025Hz,tcb10[7],'MudPy 1D SRCMOD3','o'],
                [tok_usgs_1d_025Hz,tcb10[2],   'MudPy 1D USGS',   '/'],
                [tok_src1_1d_025Hz,tcb10[0],'MudPy 1D SRCMOD1','o']],dtype=object)
funcs.plot_compare_IM_res(axes = axes[1,1],data=data,y_axis="pgd_res",ylim=[-1,2.5],
                    title='Tokachi 2003',xlabel='distance (km)',
                    tag='1d_tokachi',xticks=[50,150,250,350,450,550,650,750])

#-------------------------------------------------------------------------
# Creating legend
legend_elements = []
for i in range(5):
    legend_elements.append(Patch(alpha=0.6,edgecolor='black',linewidth=10))
    
hatches = ['o','+','o','/','x']
legend = ['MudPy 1D SRCMOD','MudPy 1D SRCMOD2','MudPy 1D SRCMOD3','MudPy 1D Hayes','MudPy 1D Zheng']
colors = [tcb10[0],tcb10[5],tcb10[7],tcb10[2],tcb10[1]]

for j, patch in enumerate(legend_elements):
    patch.set_hatch(hatches[j])
    patch.set_label(legend[j])
    patch.set_facecolor(colors[j])
    
legend_elements.append(Line2D([0],[0], linewidth=20.0, linestyle='-',  color='red', alpha=1.0,label='Zero line'))

plt.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left', 
            fontsize=130,frameon=False)

plt.text(8.3, 3.3, 'LEGEND', color='k', fontsize=130,fontdict={"weight": "bold"})
plt.text(8.3, 2.7, 'PGD Residuals (1D only)', color='k', fontsize=130)

# Add alphabet labels to the subplots
axes[0,0].text(-0.15, 1.1, "(A)", transform=axes[0,0].transAxes, fontsize=150, fontweight="bold", va="top")
axes[0,1].text(-0.15, 1.1, "(B)", transform=axes[0,1].transAxes, fontsize=150, fontweight="bold", va="top")
axes[1,0].text(-0.15, 1.1, "(C)", transform=axes[1,0].transAxes, fontsize=150, fontweight="bold", va="top")
axes[1,1].text(-0.15, 1.1, "(D)", transform=axes[1,1].transAxes, fontsize=150, fontweight="bold", va="top")



plt.show()
figpath = os.getcwd() +'/fig6.1d_pgd_residuals.png'

fig.savefig(figpath, bbox_inches='tight', dpi=100)


# Write two functions: compare mean_diff, 
#                      plot_kstest value, 
#                      plot_p_value and 
#                      plot_kstest_w_pvalue_mean_diff
#
# 1 - add profile lines

# 7 - last figures (go through different stations to look for two that shows 3D fits better)
