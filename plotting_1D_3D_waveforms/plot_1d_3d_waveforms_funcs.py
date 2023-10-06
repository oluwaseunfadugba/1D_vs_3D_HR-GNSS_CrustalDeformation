#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 13:43:33 2023

@author: oluwaseunfadugba
"""
from obspy.core import read
import matplotlib.pyplot as plt
#import os
from scipy import integrate
import pandas as pd

def plot_1d_vs_3d_waveforms(row,axes,data,sta,linewd_obs,xlim_range):
    
    
    #data = [path_z,path_n,path_e,color,tag,time_shift]
    
    fontsize = 150
    
    path_z = data[:,0]
    path_n = data[:,1]
    path_e = data[:,2]
    
    colors = data[:,3]
    tags = data[:,4]   
    time_shifts = data[:,5]
    ls = data[:,6]
    
    title_pad = 60
    
    for i in range(len(path_z)):
         
        Z_obs = read(path_z[i])
        N_obs = read(path_n[i])
        E_obs = read(path_e[i])
    
        N= len(Z_obs[0].data[time_shifts[i]:])
            
        axes[row,0].plot(Z_obs[0].times()[0:N], Z_obs[0].data[time_shifts[i]:], colors[i],\
                 label=tags[i],linewidth=linewd_obs,ls=ls[i])
        axes[row,1].plot(Z_obs[0].times()[0:N], N_obs[0].data[time_shifts[i]:], colors[i],\
                 label=tags[i],linewidth=linewd_obs,ls=ls[i])
        axes[row,2].plot(Z_obs[0].times()[0:N], E_obs[0].data[time_shifts[i]:], colors[i],\
                 label=tags[i],linewidth=linewd_obs,ls=ls[i])
            
        
  
            
            
            
            
    axes[row,0].set_title('Station '+sta+' (Z-comp)',fontsize=fontsize+15,
                        fontdict={"weight": "bold"},pad =title_pad)
    axes[row,1].set_title('Station '+sta+' (N-comp)',fontsize=fontsize+15,
                        fontdict={"weight": "bold"},pad =title_pad)
    axes[row,2].set_title('Station '+sta+' (E-comp)',fontsize=fontsize+15,
                        fontdict={"weight": "bold"},pad =title_pad)
    
    
    
    axes[row,0].xaxis.set_tick_params(labelsize=fontsize)
    axes[row,0].yaxis.set_tick_params(labelsize=fontsize)
    
    axes[row,1].xaxis.set_tick_params(labelsize=fontsize)
    axes[row,1].yaxis.set_tick_params(labelsize=fontsize)
    
    axes[row,2].xaxis.set_tick_params(labelsize=fontsize)
    axes[row,2].yaxis.set_tick_params(labelsize=fontsize)
    
    
    axes[row,0].set_xlabel('time (s)',fontsize=fontsize)
    axes[row,1].set_xlabel('time (s)',fontsize=fontsize)
    axes[row,2].set_xlabel('time (s)',fontsize=fontsize)
    
    axes[row,0].set_ylabel('displacement',fontsize=fontsize)
    
    axes[row,0].set_xlim(0,xlim_range)
    axes[row,1].set_xlim(0,xlim_range)
    axes[row,2].set_xlim(0,xlim_range)
    
    axes[row,0].tick_params(axis='x',labelsize=fontsize,labelrotation=0,length=40, width=10)
    axes[row,0].tick_params(axis='y',labelsize=fontsize,labelrotation=0,length=40, width=10)
    
    axes[row,1].tick_params(axis='x',labelsize=fontsize,labelrotation=0,length=40, width=10)
    axes[row,1].tick_params(axis='y',labelsize=fontsize,labelrotation=0,length=40, width=10)
    
    axes[row,2].tick_params(axis='x',labelsize=fontsize,labelrotation=0,length=40, width=10)
    axes[row,2].tick_params(axis='y',labelsize=fontsize,labelrotation=0,length=40, width=10)
    
    # axes[row,0].grid(color = 'k', linestyle = '-', linewidth = linewd_obs/20)
    # axes[row,1].grid(color = 'k', linestyle = '-', linewidth = linewd_obs/20)
    # axes[row,2].grid(color = 'k', linestyle = '-', linewidth = linewd_obs/20)
    
    # Increasing the linewidth of the frame border 
    for pos in ['right', 'top', 'bottom', 'left']:
        axes[row,0].spines[pos].set_linewidth(linewd_obs/3)
        axes[row,1].spines[pos].set_linewidth(linewd_obs/3)
        axes[row,2].spines[pos].set_linewidth(linewd_obs/3)
        
        
        
    
            
    return     
            
            
    # # ------------------------------------------------------------------------------------------
    # # Z-component
    # ax1 = fig.add_subplot(1, 3, 1)
    # ax1.set_title('Station '+ str(sta) + ' (Z-comp)', fontsize=fontsize)
    # #ax1.plot(Z_obs[0].times()[0:(xlim_range+1)], Z_obs[0].data[time_shift:(xlim_range+1+time_shift)], "k-",\
    # N= len(Z_obs[0].data[time_shift:])
    # ax1.plot(Z_obs[0].times()[0:N], Z_obs[0].data[time_shift:], "k-",\
    #          label='Observed',linewidth=linewd_obs)
    # ax1.xaxis.set_tick_params(labelsize=fontsize)
    # ax1.yaxis.set_tick_params(labelsize=fontsize)
    # ax1.set_xlabel('time (s)',fontsize=fontsize)
    # ax1.set_ylabel('displacement',fontsize=fontsize)
    # ax1.set_xlim(0,xlim_range)
    # ax1.legend(fontsize=fontsize-5)
    # plt.grid()

    # # ------------------------------------------------------------------------------------------
    # # N-component
    # ax2 = fig.add_subplot(1, 3, 2)
    # # ax2.set_title('Station '+ str(sta) + ' (N-comp)', fontsize=fontsize)
    # # #ax2.plot(N_obs[0].times()[0:(xlim_range+1)], N_obs[0].data[time_shift:(xlim_range+1+time_shift)], "k-",\
    # # N = len(N_obs[0].data[time_shift:])
    # # ax2.plot(N_obs[0].times()[0:N], N_obs[0].data[time_shift:], "k-",\
    # #      label='Observed',linewidth=linewd_obs)
    # # ax2.xaxis.set_tick_params(labelsize=fontsize)
    # # ax2.yaxis.set_tick_params(labelsize=fontsize)
    # # ax2.set_xlabel('time (s)',fontsize=fontsize)
    # # ax2.set_ylabel('displacement',fontsize=fontsize)
    # # ax2.set_xlim(0,xlim_range)
    # # ax2.legend(fontsize=fontsize-5)
    # # plt.grid()

    # # # ---------------------------------------------
    # # E-component
    # ax3 = fig.add_subplot(1, 3, 3)
    # # ax3.set_title('Station '+ str(sta) + ' (E-comp)', fontsize=fontsize)
    # # #ax3.plot(E_obs[0].times()[0:(xlim_range+1)], E_obs[0].data[time_shift:(xlim_range+1+time_shift)], "k-",\
    # # N = len( E_obs[0].data[time_shift:])
    # # ax3.plot(E_obs[0].times()[0:N], E_obs[0].data[time_shift:], "k-",\
    # #          label='Observed',linewidth=linewd_obs)
    # # ax3.xaxis.set_tick_params(labelsize=fontsize)
    # # ax3.yaxis.set_tick_params(labelsize=fontsize)
    # # ax3.set_xlabel('time (s)',fontsize=fontsize)
    # # ax3.set_ylabel('displacement',fontsize=fontsize)
    # # ax3.set_xlim(0,xlim_range)
    # # ax3.legend(fontsize=fontsize-5)
    # # plt.grid()

    # return ax1, ax2, ax3
