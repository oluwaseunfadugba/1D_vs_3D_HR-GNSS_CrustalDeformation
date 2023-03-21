#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 16:30:01 2023

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



#%% Creating the Fig S8 
#-------Ibaraki 2011-----
fig, axes = plt.subplots(4,2,figsize=(70, 100))
fig.tight_layout(h_pad=60,w_pad=40)#, w_pad=0.5, h_pad=7.0,rect=[0, 0.15, 0.8, 0.9]) #0, 0.03, 0.85, 0.90]

#%% 
models = np.array([[iba_src_1d_025Hz,   iba_src_3d_025Hz_tal,  'ibaraki_srcmod',     [0], [5,9]],
                    [iba_zh_1d_025Hz,   iba_zh_3d_025Hz_tal,   'ibaraki_zheng',      [0], [0,1]],
                    [iwa_zh_1d_025Hz,   iwa_zh_3d_025Hz_las,   'iwate_zheng_lasson', [0], [0,1]],
                    [miy_usgs_1d_025Hz, miy_usgs_3d_025Hz_las, 'miyagi_usgs_lasson', [0], [0,1]],
                    [miy_zh_1d_025Hz, miy_zh_3d_025Hz_las,   'miyagi_zheng_lasson',  [0], [0,1]],
                    [tok_src3_1d_025Hz, tok_src3_3d_025Hz_tal, 'tokachi_srcmod3',    [0], [1,4]],
                    [tok_usgs_1d_025Hz, tok_usgs_3d_025Hz_tal, 'tokachi_usgs',       [0], [0,1]]],dtype=object)


kstest_ff_path_='/Users/oluwaseunfadugba/Documents/Projects/TsE_ValerieDiego/'+\
    'TsE_1D_vs_3D/Paper_Figures/Comparing_kstest_Residuals_with_median_diff/'
    
xticks = [0,500,1000,1500]

# plotting mean difference residuals
funcs.plot_median_residual(axes=axes[0,0],models=models,attribute='pgd_res',
                            xticks=xticks,yticks=[1,0,-1,-2],subplt_label='A',
                            kstest_ff_path=kstest_ff_path_)

funcs.plot_median_residual(axes=axes[1,0],models=models,attribute='tPGD_res',
                            xticks=xticks,yticks=[40,20,0,-20],subplt_label='C',
                            kstest_ff_path=kstest_ff_path_,ylabel='s')

funcs.plot_median_residual(axes=axes[2,0],models=models,attribute='sd_res',
                            xticks=xticks,yticks=[2,0,-2],subplt_label='E',
                            kstest_ff_path=kstest_ff_path_)

funcs.plot_median_residual(axes=axes[3,0],models=models,attribute='xcorr',rec_neg=0,
                            xticks=xticks,yticks=[0.1,0,-0.1],subplt_label='G',ylabel='value',
                            kstest_ff_path=kstest_ff_path_,xlabel='distance (km)')


# plotting p-value
funcs.plot_median_residual(axes=axes[0,1],models=models,attribute='pgd_res',
                            xticks=xticks,yticks=[1,0.5,0],subplt_label='B',
                            kstest_ff_path=kstest_ff_path_,pvalue = 1)

funcs.plot_median_residual(axes=axes[1,1],models=models,attribute='tPGD_res',
                            xticks=xticks,yticks=[1,0.5,0],subplt_label='D',
                            kstest_ff_path=kstest_ff_path_,pvalue = 1)

funcs.plot_median_residual(axes=axes[2,1],models=models,attribute='sd_res',
                            xticks=xticks,yticks=[1,0.5,0],subplt_label='F',
                            kstest_ff_path=kstest_ff_path_,pvalue = 1)

funcs.plot_median_residual(axes=axes[3,1],models=models,attribute='xcorr',
                            xticks=xticks,yticks=[1,0.5,0],subplt_label='H',
                            kstest_ff_path=kstest_ff_path_,xlabel='distance (km)',pvalue = 1)


# #-------------------------------------------------------------------------
# Creating legend
legend_elements = []

legend_elements.append(Line2D([0],[0], linewidth=20.0, linestyle=ls[0], color=tcb10[0], 
                              alpha=1.0,label='Ibaraki 2011 SRCMOD'))

legend_elements.append(Line2D([0],[0], linewidth=20.0, linestyle=ls[1], color=tcb10[1], 
                              alpha=1.0,label='Ibaraki 2011 Zheng'))

legend_elements.append(Line2D([0],[0], linewidth=20.0, linestyle=ls[2], color=tcb10[2], 
                              alpha=1.0,label='Iwate 2011 Zheng'))

legend_elements.append(Line2D([0],[0], linewidth=20.0, linestyle=ls[3], color=tcb10[3], 
                              alpha=1.0,label='Miyagi 2011 Hayes'))

legend_elements.append(Line2D([0],[0], linewidth=20.0, linestyle=ls[4], color=tcb10[4], 
                              alpha=1.0,label='Miyagi 2011 Zheng'))

legend_elements.append(Line2D([0],[0], linewidth=20.0, linestyle=ls[5], color=tcb10[5], 
                              alpha=1.0,label='Tokachi 2003 SRCMOD3'))

legend_elements.append(Line2D([0],[0], linewidth=20.0, linestyle=ls[6], color=tcb10[6], 
                              alpha=1.0,label='Tokachi 2003 Hayes'))


legend_elements.append(Line2D([0],[0], linewidth=20.0, linestyle='--', color='black', 
                              alpha=1.0,label='$\delta_{|3D|-|1D|}$ = 0'))

legend_elements.append(Line2D([0],[0], linewidth=20.0, linestyle='--', color='black', 
                              alpha=1.0,label='p-value = 0'))

plt.legend(handles=legend_elements, bbox_to_anchor=(1.17, 5.0 ), loc='upper left', 
            fontsize=160,frameon=False,labelspacing = 1)

# Insert texts
plt.text(2000.0, 5.6, 'LEGEND', color='k', fontsize=160,fontdict={"weight": "bold"})
plt.text(-600, 6.6, 'PGD Residuals', color='k', fontsize=160)
plt.text(-600, 4.75, 'tPGD Residuals', color='k', fontsize=160)
plt.text(-600, 3, 'SD Residuals', color='k', fontsize=160)
plt.text(-600, 1.25, 'Xcorr Residuals', color='k', fontsize=160)

# Adding mean difference residual cartoon figure
im = plt.imread(get_sample_data(os.getcwd()+'/mean_diff_cartoon.png'))
newax = fig.add_axes([1.08, -0.1, 0.5, 0.5], anchor='NE', zorder=-1)
newax.imshow(im)
newax.axis('off')


plt.show()

figpath = os.getcwd() +'/fig10.mean_diff_and_pvalue.png'

fig.savefig(figpath, bbox_inches='tight', dpi=100)