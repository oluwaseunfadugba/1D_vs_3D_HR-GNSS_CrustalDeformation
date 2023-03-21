#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  9 12:10:17 2022

@author: oluwaseunfadugba
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import time

start = time.time()

path = '/Users/oluwaseunfadugba/Documents/Projects/TsE_ValerieDiego/'+\
    'TsE_1D_vs_3D/1D_vs_3D_HR-GNSS_CrustalDeformation/velocity_models/'
    
os.chdir(path)

np.set_printoptions(suppress=True) 

#%% Utility functions
def write_mod(vel_model,no_layer,outmod):
    # vel_model is in th,vs,vp,rho,qs,qp
    if no_layer=='all': 
        no_layer = vel_model[:,0].size
        
    print(no_layer)
   
    f = open(outmod, "w")
    for i in range(no_layer):
        if i != no_layer-1:
            th = vel_model[i+1,0]-vel_model[i,0]
            
            f.write(str(th) + "\t" + str(vel_model[i,2]) + "\t" + 
                    str(vel_model[i,1]) + "\t" + str(vel_model[i,3]) + "\t" + 
                    str(vel_model[i,5]) + "\t" + str(vel_model[i,4]) +"\n")
        else:
            f.write(str("0") + "\t" + str(vel_model[i,2]) + "\t" + 
                    str(vel_model[i,1]) + "\t" + str(vel_model[i,3]) + "\t" + 
                    str(vel_model[i,5]) + "\t" + str(vel_model[i,4]) +"\n")
                
    f.close()   
    return

def plot_mod(mod,velname,outfig):
    
    data = np.genfromtxt(mod)
    data[:,0] = data[:,0].cumsum()
    data[-1,0] = data[-1,0]+10
    l = len(data[:,0])
    
    vp = [data[:,2][i//2]*1e3 for i in range(l*2)]
    vs = [data[:,1][i//2]*1e3 for i in range(l*2)]
    rho = [data[:,3][i//2]*1e3 for i in range(l*2)]
    qs = [data[:,4][i//2] for i in range(l*2)]
    qp = [data[:,5][i//2] for i in range(l*2)]
    #print(data)
    
    depth = [0]
    for i in data[:,0]:
        depth.extend([i, i+0.0001])
    depth = depth[:-1]
    
    # plot velocity model
    linewidth = 7
    fontsize = 60
    fig = plt.figure(figsize=(20, 20))
    plt.plot(vp,depth, 'r--',label="Vp (m/s)",linewidth=linewidth)
    plt.plot(vs, depth, 'b-.',label="Vs (m/s)",linewidth=linewidth)
    plt.plot(rho, depth, 'g-',label="den (g/cm3)",linewidth=linewidth)
    plt.plot(qs, depth, 'c-',label="Qs",linewidth=linewidth)
    plt.plot(qp, depth, 'm--',label="Qp",linewidth=linewidth)

    plt.title('1D Velocity Model %s' % velname,fontsize=70, pad=50)
    plt.xlabel('Material properties',fontsize=fontsize)
    plt.ylabel('depth (km)',fontsize=fontsize)
    
    plt.gca().xaxis.set_tick_params(labelsize=fontsize)
    plt.gca().yaxis.set_tick_params(labelsize=fontsize)
    
    print(depth[-1])
    plt.ylim((0,depth[-1]))
    lgd = plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
          fancybox=False, shadow=True, ncol=3,fontsize=fontsize-10) #, 
    plt.grid(color='k', linestyle='-', linewidth=0.5)
    plt.gca().invert_yaxis()
    #plt.xaxis.set_label_position('top')
    plt.show()
    fig.savefig(outfig,bbox_extra_artists=(lgd,), bbox_inches='tight',dpi=100,facecolor='white')
    
    return

#%%
# Making .mod for Hayes (2017)
velname = '(Hayes, 2017)'
outmod = path+'1d_vel_japan.mod'
outfig = path+'fig3.1d_vel_japan.png'
no_layer = 5 # 'all' for all layers

# DEPTH[km]	P-VEL[km/s] S-VEL[km/s] DENS[g/cm^3]	QP QS
vel_model = np.array([\
[0.00,  2.50,  1.20,	 2.10, 1000.00, 500.00],
[1.00,	6.00,  3.40,	 2.70, 1000.00, 500.00],
[11.00, 6.60,  3.70,	 2.90, 1000.00, 500.00],
[21.00,	7.20,  4.00,	 3.10, 1000.00, 500.00],
[31.00,	8.08,  4.47,	 3.38, 1200.00, 500.00],
[227.0,	8.59,  4.66,	 3.45,  360.00, 140.00]])
    
print(vel_model)

# Write vel_model to file
write_mod(vel_model,no_layer,outmod)

# Plot .mod
plot_mod(outmod,velname,outfig)

    
# ####################################################################
end = time.time()
time_elaps = end - start
if time_elaps < 60:
    print(f'Duration: {round(time_elaps)} seconds')
else:
    print(f'Duration: {round(time_elaps/60)} minutes')  
  