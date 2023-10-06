#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 13:27:42 2023

@author: oluwaseunfadugba
"""


import os
wkpath = '/Users/oluwaseunfadugba/Documents/Projects/TsE_ValerieDiego/TsE_1D_vs_3D/1D_vs_3D_HR-GNSS_CrustalDeformation/plotting_1D_3D_waveforms/'
os.chdir(wkpath)
import plot_1d_3d_waveforms_funcs as funcs 
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
import time
from string import ascii_lowercase as alphab

start = time.time()

#%% ibaraki2011_srcmod_srf3d_rupt1_fmax_0.15
# Observed waveforms  
home_obs = '/Users/oluwaseunfadugba/Documents/Projects/TsE_ValerieDiego/TsE_1D_vs_3D/3D_Modeling_using_SW4/Running_Simulations/4_Model_results/'
p_obs_Ibaraki2011_025Hz = home_obs+'All_025Hz_Result/ibaraki2011_srcmod_srf3d_rupt5_talapas/'
p_obs_Ibaraki2011_050Hz = home_obs+'All_05Hz_Results/ibaraki2011_srcmod_srf3d_rupt5/'
    
# MudPy 1D
home_1d = '/Users/oluwaseunfadugba/Documents/Projects/TsE_ValerieDiego/'+\
    'TsE_1D_vs_3D/1D_Modeling_using_FQs_Mudpy/Running_FakeQuakes_now/ibaraki2011_srcmod/'
p_1d_Ibaraki2011_025Hz = home_1d+'waveforms_0.25Hz/'
p_1d_Ibaraki2011_050Hz = home_1d+'waveforms_0.49Hz/'
   
#SW4_3D
home_3d = '/Users/oluwaseunfadugba/Documents/Projects/TsE_ValerieDiego/TsE_1D_vs_3D/3D_Modeling_using_SW4/Running_Simulations/4_Model_results/'
p_3d_Ibaraki2011_025Hz = home_3d+'All_025Hz_Result/ibaraki2011_srcmod_srf3d_rupt5_talapas/'
p_3d_Ibaraki2011_050Hz = home_3d+'All_05Hz_Results/ibaraki2011_srcmod_srf3d_rupt5/'

#%%
tcb10 = ['#006BA4', '#FF800E', '#ABABAB', '#595959', '#5F9ED1', '#C85200', '#898989', '#A2C8EC', '#FFBC79', '#CFCFCF']
plt.style.use('tableau-colorblind10') #('seaborn-colorblind') #

xlim_range = 370
linewd_obs = 20

fig, axes = plt.subplots(2,3,figsize=(120, 75))
fig.tight_layout(h_pad=80,w_pad=70)
plt.text(-710, 0.69, 'Ibaraki 2011 (SRCMOD Rupt 5)', color='k', fontsize=250,fontdict={"weight": "bold"})


# Rupt 5
sta = ['0041','0043']   #,'0042','0043'
for i in range(len(sta)):
    
    iba_obs_025Hz_z = p_obs_Ibaraki2011_025Hz+'ibaraki2011_srcmod_srf3d_rupt5.sw4output_z_obs.0.25Hz/'+sta[i]+'.obs_z.sac.0.25Hz'
    iba_obs_025Hz_n = p_obs_Ibaraki2011_025Hz+'ibaraki2011_srcmod_srf3d_rupt5.sw4output_n_obs.0.25Hz/'+sta[i]+'.obs_n.sac.0.25Hz'
    iba_obs_025Hz_e = p_obs_Ibaraki2011_025Hz+'ibaraki2011_srcmod_srf3d_rupt5.sw4output_e_obs.0.25Hz/'+sta[i]+'.obs_e.sac.0.25Hz'
    
    iba_1d_025Hz_z = p_1d_Ibaraki2011_025Hz+'ibaraki2011_srcmod.000005_z_syn.0.25Hz/'+sta[i]+'.syn_z.sac.0.25Hz'
    iba_1d_025Hz_n = p_1d_Ibaraki2011_025Hz+'ibaraki2011_srcmod.000005_n_syn.0.25Hz/'+sta[i]+'.syn_n.sac.0.25Hz'
    iba_1d_025Hz_e = p_1d_Ibaraki2011_025Hz+'ibaraki2011_srcmod.000005_e_syn.0.25Hz/'+sta[i]+'.syn_e.sac.0.25Hz'
    
    iba_1d_050Hz_z = p_1d_Ibaraki2011_050Hz+'ibaraki2011_srcmod.000005_z_syn.0.49Hz/'+sta[i]+'.syn_z.sac.0.49Hz'
    iba_1d_050Hz_n = p_1d_Ibaraki2011_050Hz+'ibaraki2011_srcmod.000005_n_syn.0.49Hz/'+sta[i]+'.syn_n.sac.0.49Hz'
    iba_1d_050Hz_e = p_1d_Ibaraki2011_050Hz+'ibaraki2011_srcmod.000005_e_syn.0.49Hz/'+sta[i]+'.syn_e.sac.0.49Hz'
    
    
    iba_3d_025Hz_z = p_3d_Ibaraki2011_025Hz+'ibaraki2011_srcmod_srf3d_rupt5.sw4output_z_syn.0.25Hz/'+sta[i]+'.syn_z.sac.0.25Hz'
    iba_3d_025Hz_n = p_3d_Ibaraki2011_025Hz+'ibaraki2011_srcmod_srf3d_rupt5.sw4output_n_syn.0.25Hz/'+sta[i]+'.syn_n.sac.0.25Hz'
    iba_3d_025Hz_e = p_3d_Ibaraki2011_025Hz+'ibaraki2011_srcmod_srf3d_rupt5.sw4output_e_syn.0.25Hz/'+sta[i]+'.syn_e.sac.0.25Hz'
    
    iba_3d_050Hz_z = p_3d_Ibaraki2011_050Hz+'ibaraki2011_scrmod_srf3d_rupt5.sw4output_z_syn.0.49Hz/'+sta[i]+'.syn_z.sac.0.49Hz'
    iba_3d_050Hz_n = p_3d_Ibaraki2011_050Hz+'ibaraki2011_scrmod_srf3d_rupt5.sw4output_n_syn.0.49Hz/'+sta[i]+'.syn_n.sac.0.49Hz'
    iba_3d_050Hz_e = p_3d_Ibaraki2011_050Hz+'ibaraki2011_scrmod_srf3d_rupt5.sw4output_e_syn.0.49Hz/'+sta[i]+'.syn_e.sac.0.49Hz'
    
    # [path_z,path_n,path_e,color,tag,time_shift]
    data =np.array([[iba_obs_025Hz_z,iba_obs_025Hz_n,iba_obs_025Hz_e,tcb10[3],'obs_025Hz',20,'-'],
                    #[iba_1d_025Hz_z,iba_1d_025Hz_n,iba_1d_025Hz_e,tcb10[1],'Mudpy_025Hz',0],
                    [iba_1d_050Hz_z,iba_1d_050Hz_n,iba_1d_050Hz_e,tcb10[2],'Mudpy_050Hz',0,'--'],
                    [iba_3d_050Hz_z,iba_3d_050Hz_n,iba_3d_050Hz_e,tcb10[0],'SW4_050Hz',0,'-'],
                    [iba_3d_025Hz_z,iba_3d_025Hz_n,iba_3d_025Hz_e,tcb10[1],'SW4_025Hz',0,'-.']],dtype=object)

    funcs.plot_1d_vs_3d_waveforms(i,axes,data,sta[i],linewd_obs,xlim_range)


# Adding legend
legend_elements = []

legend = ['Observed','MudPy 1D','SW4 3D (fmax = 0.25 Hz)','SW4 3D (fmax = 0.50 Hz)']
colors = [tcb10[3],tcb10[2],tcb10[1],tcb10[0]]

legend_elements.append(Line2D([0],[0], linewidth=linewd_obs, linestyle='-',  color=colors[0], 
                              alpha=1.0,label=legend[0]))

legend_elements.append(Line2D([0],[0], linewidth=linewd_obs, linestyle='--',  color=colors[1], 
                              alpha=1,label=legend[1]))

legend_elements.append(Line2D([0],[0], linewidth=linewd_obs, linestyle='-.',  color=colors[2], 
                              alpha=1.0,label=legend[2]))

legend_elements.append(Line2D([0],[0], linewidth=linewd_obs, linestyle='-',  color=colors[3], 
                              alpha=1.0,label=legend[3]))

plt.legend(handles=legend_elements, bbox_to_anchor=(-2, -0.6 ), loc='lower left', 
            fontsize=160,frameon=False, ncol=2)

plt.text(-720, -0.1, 'LEGEND', color='k', fontsize=170,fontdict={"weight": "bold"})

# subplot label
ax = axes.flatten()
subplt_labelpos=[-0.2, 1.1]
for j in range(6):
    # Add alphabet labels to the subplots
    ax[j].text(subplt_labelpos[0], subplt_labelpos[1], '('+alphab[j].upper()+')', 
              transform=ax[j].transAxes, fontsize=160, fontweight="bold", va="top")




plt.savefig('fig7.1d_3d_waveforms_Ibaraki2011_srcmod_rupt5.png', bbox_inches='tight', dpi=100)

plt.show()

# ####################################################################
end = time.time()
time_elaps = end - start
if time_elaps < 60:
    print(f'Duration: {round(time_elaps)} seconds')
else:
    print(f'Duration: {round(time_elaps/60)} minutes')
