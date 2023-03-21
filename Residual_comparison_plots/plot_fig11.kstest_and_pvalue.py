#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 19:09:07 2023

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
ls = ['-','--','-.','-','--','-.','-','--','-.','-','--']

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



#%% Kstest and pvalue results
fig, axes = plt.subplots(2,2,figsize=(90, 70))
fig.tight_layout(h_pad=70,w_pad=110)#, w_pad=0.5, h_pad=7.0,rect=[0, 0.15, 0.8, 0.9]) #0, 0.03, 0.85, 0.90]


xticks = np.array([['ibaraki_srcmod',      'IB_S'],
                    ['ibaraki_zheng',      'IB_Z'],
                    ['iwate_zheng_lasson', 'IW_Z'],
                    ['miyagi_usgs_lasson', 'MI_H'],
                    ['miyagi_zheng_lasson','MI_Z'],
                    ['tokachi_srcmod3',    'TO_S'],
                    ['tokachi_usgs',       'TO_H']],dtype=object)


kstest_ff_path_='/Users/oluwaseunfadugba/Documents/Projects/TsE_ValerieDiego/'+\
    'TsE_1D_vs_3D/Paper_Figures/Comparing_kstest_Residuals_with_median_diff/'
    

# plotting mean difference residuals
funcs.plot_ks_test_all_w_median(axes=axes[0,0],attribute='pgd_res',title_pad = 50,
                            subplt_label='A',xticks=xticks,title='PGD Residual',
                            kstest_ff_path=kstest_ff_path_,ylabel='ln')

funcs.plot_ks_test_all_w_median(axes=axes[0,1],attribute='tPGD_res',title_pad = 50,
                            subplt_label='B',xticks=xticks,title='tPGD Residual',
                            kstest_ff_path=kstest_ff_path_,ylabel='s')

funcs.plot_ks_test_all_w_median(axes=axes[1,0],attribute='sd_res',title_pad = 50,
                            subplt_label='C',xticks=xticks,title='SD Residual',
                            kstest_ff_path=kstest_ff_path_,xlabel='Simulations',ylabel='ln')

funcs.plot_ks_test_all_w_median(axes=axes[1,1],attribute='xcorr',title_pad = 50,
                            subplt_label='D',xticks=xticks,title='Xcorr',ylabel='value',
                            kstest_ff_path=kstest_ff_path_,xlabel='Simulations')


# #-------------------------------------------------------------------------
# Creating legend
legend_elements = []


legend_elements.append(Line2D([0],[0], linewidth=20.0, linestyle=ls[0], color=tcb10[0], 
                              alpha=1.0,label='ks-stat'))

legend_elements.append(Line2D([0],[0], linewidth=20.0, linestyle=ls[0], color=tcb10[0], 
                              alpha=1.0,label='D Critical'))

legend_elements.append(Line2D([0],[0], linewidth=20.0, linestyle=ls[1], color=tcb10[1], 
                              alpha=1.0,label='p-value'))

legend_elements.append(Line2D([0],[0], linewidth=20.0, linestyle=ls[1], color=tcb10[1], 
                              alpha=1.0,label='p-val = 0.05'))

legend_elements.append(Line2D([0],[0], linewidth=20.0, linestyle=ls[0], color=tcb10[2], 
                              alpha=1.0,label='$\delta_{|3D|-|1D|}$ '))

# legend_elements.append(Line2D([0],[0], linewidth=20.0, linestyle=ls[0], color=tcb10[2], 
#                               alpha=1.0,label='$n_{1D}+n_{3D}$ = 2930'))

legend_elements.append(Line2D([],[],color=tcb10[0], marker='o', markerfacecolor=tcb10[0],
                              markersize=100,alpha=1.0,label='$n_{1D}+n_{3D}$ = 2930'))

legend_elements.append(Line2D([0],[0],color=tcb10[0], marker='o', markerfacecolor=tcb10[0],
                              markersize=40,alpha=1.0,label='$n_{1D}+n_{3D}$ = 804'))



# legend_elements.append(Line2D([0],[0], linewidth=20.0, linestyle=ls[0], color=tcb10[2], 
#                               alpha=1.0,label='$n_{1D}+n_{3D}$ = 804'))


#line1 = Line2D([], [], color="white", marker='o', markerfacecolor="red")




plt.legend(handles=legend_elements, bbox_to_anchor=(1.3, 2.2 ), loc='upper left', 
            fontsize=160,frameon=False,labelspacing = 1)



# Insert texts
plt.text(8.5, 0.23, 'LEGEND', color='k', fontsize=160,fontdict={"weight": "bold"})

plt.text(8.5,0.06,   'Simulations', color='k', fontsize=160)
plt.text(8.5, 0.04,  'IB_S - Ibaraki 2011 SRCMOD', color='k', fontsize=160)
plt.text(8.5, 0.02,  'IB_Z - Ibaraki 2011 Zheng', color='k', fontsize=160)
plt.text(8.5, 0.0,   'IW_Z - Iwate 2011 Zheng', color='k', fontsize=160)
plt.text(8.5, -0.02, 'MI_H - Miyagi 2011 Hayes', color='k', fontsize=160)
plt.text(8.5, -0.04, 'MI_Z - Miyagi 2011 Zheng', color='k', fontsize=160)
plt.text(8.5, -0.06, 'TO_S - Tokachi 2003 SRCMOD3', color='k', fontsize=160)
plt.text(8.5, -0.08, 'TO_H - Tokachi 2003 Hayes', color='k', fontsize=160)





plt.show()

figpath = os.getcwd() +'/fig11.kstest_all_and_pvalue.png'

fig.savefig(figpath, bbox_inches='tight', dpi=100)
