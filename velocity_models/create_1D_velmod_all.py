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

#%%  Making .mod for Hayes (2017)
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

#%%  Making .mod for VJMA2011
velname = '(VJMA, 2011)'
outmod = path+'1d_vel_japan_vjma.mod'
outfig = path+'fig.1v_vel_japan_vjma.png'
no_layer = 'all' #1880 # [] for all layers
    
# Prepare vel_model [DEPTH[km] P-VEL[km/s] S-VEL[km/s] DENS[g/cm^3]	QP QS]
data = np.genfromtxt("vjma2001")

vp = data[:,0]
vs = data[:,1]
depth = data[:,2]

# determine rho, qp and qs
vel_model = np.ones([vp.size,6])
rho = (1.6612 * vp - 0.4721 * vp**2 + 0.0671 * vp**3
        - 0.0043 * vp**4 + 0.000106 * vp**5)
qs = (-16 + 104.13 * vs - 25.225 * vs**2 + 8.2184 * vs**3)
qs[vs < 0.3] = 13
qp = 2 * qs

vel_model[:,0] = depth
vel_model[:,1] = vp
vel_model[:,2] = vs
vel_model[:,3] = rho
vel_model[:,4] = qp
vel_model[:,5] = qs

print(vel_model)

# Write vel_model to file
write_mod(vel_model,no_layer,outmod)

# # Plot .mod
plot_mod(outmod,velname,outfig)

#%% Making .mod for Yagi (2004)

velname = '(Yagi, 2004)'
outmod = path+'1d_vel_japan_yagi_vel.mod'
outfig = path+'fig.1d_vel_japan_yagi_vel.png'
no_layer = 'all' # 'all' for all layers

# DEPTH[km]	P-VEL[km/s] S-VEL[km/s] DENS[g/cm^3]	QP QS
vel_model = np.array([\
[0.00, 3.8,  2.19,	 2.3, 300.00, 150.00],
[4.00,	5.5,  3.18,	 2.6, 500.00, 250.00],
[8.00, 5.8,  3.34,	 2.7, 500.00, 250.00],
[18.00,	6.5,  3.74,	 2.9, 600.00, 300.00],
[28.00,	7.8,  4.5,	 3.2, 1200.00, 600.00]])
    
print(vel_model)

# Write vel_model to file
write_mod(vel_model,no_layer,outmod)

# Plot .mod
plot_mod(outmod,velname,outfig)


#%% Making .mod for Zheng et al (2020)

velname = '(Zheng etal, 2020)'
outmod = path+'1d_vel_japan_zheng_vel.mod'
outfig = path+'fig.1d_vel_japan_zheng_vel.png'
no_layer = 'all' # 'all' for all layers

# DEPTH[km]	P-VEL[km/s] S-VEL[km/s] DENS[g/cm^3]	QP QS
vel_model = np.array([\
[0.00, 6.0,  3.5,	 2.72, 0, 0],
[2.05,	6.0,  3.5,	 2.86, 0, 0],
[6.04, 7.1,  3.9,	 3.05, 0, 0],
[12.03,	7.9,  4.4,	 3.26, 0, 0],
[120.0,	8.05,  4.6,	 3.45, 0, 0],
[165.0,	8.18,  4.72, 3.47, 0, 0],
[210.0,	8.30,  4.79, 3.52, 0, 0],  # still need to edit depth, vp, vs, rho
[260.0,	8.48,  5.0,	 3.57, 0, 0],
[310.0,	8.67,  5.11, 3.61, 0, 0],
[360.0,	8.85,  5.31, 3.66, 0, 0],
[410.0,	9.03,  5.21, 3.71, 0, 0],
[460.0,	9.53,  5.5,	 3.93, 0, 0]])
    
print(vel_model)
vs = vel_model[:,2]

qs = (-16 + 104.13 * vs - 25.225 * vs**2 + 8.2184 * vs**3)
qs[vs < 0.3] = 13
qp = 2 * qs

vel_model[:,4] = np.round(qp,2)
vel_model[:,5] = np.round(qs,2)

print(vel_model)

# Write vel_model to file
write_mod(vel_model,no_layer,outmod)

# Plot .mod
plot_mod(outmod,velname,outfig)

#%% Making .mod for Koketsu et al (2004)

velname = '(Koketsu etal, 2004)'
outmod = path+'1d_vel_japan_kok_vel.mod'
outfig = path+'fig.1d_vel_japan_kok_vel.png'
no_layer = 'all' # 'all' for all layers

# DEPTH[km]	P-VEL[km/s] S-VEL[km/s] DENS[g/cm^3]	QP QS
vel_model = np.array([\
[0.00, 5.8,  3.3,	 0, 0, 0],
[1.5,	5.8,  3.3,	 0, 0, 0],
[4.0, 6.1,  3.5,	 0, 0, 0],
[14.0,	6.5,  3.7,	 0, 0, 0],
[20.0,	7.0,  4.0,	 0, 0, 0],
[28.0,	7.7,  4.4,   0, 0, 0],
[60.0,	8.0,  4.6,   0, 0, 0]])
    
print(vel_model)

vp = vel_model[:,1]
vs = vel_model[:,2]

rho = (1.6612 * vp - 0.4721 * vp**2 + 0.0671 * vp**3
        - 0.0043 * vp**4 + 0.000106 * vp**5)
qs = (-16 + 104.13 * vs - 25.225 * vs**2 + 8.2184 * vs**3)
qs[vs < 0.3] = 13
qp = 2 * qs

vel_model[:,3] = np.round(rho,2)
vel_model[:,4] = np.round(qp,2)
vel_model[:,5] = np.round(qs,2)

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
  