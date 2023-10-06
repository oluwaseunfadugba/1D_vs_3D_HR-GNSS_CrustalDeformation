#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 17 12:33:36 2022

@author: oluwaseunfadugba
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 23 14:54:20 2022

@author: oluwaseunfadugba
"""

# Plotting the STF
# Plotting the point source locations for Ibaraki2011_SRCMOD.00000
import os
os.sys.path.insert(0, "/Users/oluwaseunfadugba/code/MudPy/src/python")

import numpy as np
from numpy import genfromtxt,unique,where,zeros
import matplotlib.pyplot as plt
from mudpy import view #fakequakes,runslip,forward,
import time
start = time.time()

cwd = '/Users/oluwaseunfadugba/Documents/Projects/TsE_ValerieDiego/TsE_1D_vs_3D/1D_vs_3D_HR-GNSS_CrustalDeformation/rupture_figs/'
os.chdir(cwd)

from matplotlib import cm
new_cmap = cm.get_cmap('rainbow', 9)#OrRd or rainbow


#%% extract_rupt_data
def extract_rupt_data(rupt,slip_percent =-10):
    f=genfromtxt(rupt)
    num=f[:,0]
    all_ss=f[:,8]
    all_ds=f[:,9]
    
    #Now parse for multiple rupture speeds
    unum=unique(num)
    ss=zeros(len(unum))
    ds=zeros(len(unum))
    for k in range(len(unum)):
        i=where(unum[k]==num)
        ss[k]=all_ss[i].sum()
        ds[k]=all_ds[i].sum()
    #Sum them
    slip=(ss**2+ds**2)**0.5
    #Get other parameters
    lon=f[0:len(unum),1]
    lat=f[0:len(unum),2]
    strike=f[0:len(unum),4]
    #Get projection of rake vector
    x,y=view.slip2geo(ss,ds,strike)
    
    #keep only appropriate slip
    i=where(slip > slip_percent*slip.max())
    slip=slip[i]
    lon=lon[i]
    lat=lat[i]
    x=x[i]
    y=y[i]
    
    return lon,lat,x,y,slip

#%% plot_4srf
def plot_4srf(rupt_name,rupt1,rupt2,rupt3,rupt4,max_slip,title):
    lon1,lat1,x1,y1,slip1 = extract_rupt_data(rupt1,slip_percent =-10)
    lon2,lat2,x2,y2,slip2 = extract_rupt_data(rupt2,slip_percent =-10)
    lon3,lat3,x3,y3,slip3 = extract_rupt_data(rupt3,slip_percent =-10)
    lon4,lat4,x4,y4,slip4 = extract_rupt_data(rupt4,slip_percent =-10)
    
    slip1[slip1>max_slip]=max_slip-.01
    slip2[slip2>max_slip]=max_slip-.01
    slip3[slip3>max_slip]=max_slip-.01
    slip4[slip4>max_slip]=max_slip-.01
    
    #rupt_name = 'Ibaraki 2011, SRCMOD'
    
    fig = plt.figure(figsize=(50,15), dpi=100)
    fig.suptitle(title, fontsize=70, y=1.05)
    
    levels = np.linspace(0, max_slip, max_slip+1)
    
    size_title = 45
    xlabel_size = 40
    ms = 9
    grd_w = 1.5
    rot = 0
   
    ax1 = fig.add_subplot(1, 4,1)
    ax1.tricontour(lon1,lat1, slip1, levels=levels, linewidths=0.01, colors='k')
    cntr2 = ax1.tricontourf(lon1,lat1, slip1, levels=levels, cmap=new_cmap) #RdBu_r
    ax1.plot(lon1,lat1, 'ko', ms=ms)
    ax1.set_title('Mean Rupture Model',fontsize=size_title)
    ax1.set_xlabel('Longitude',fontsize=xlabel_size)
    ax1.set_ylabel('Latitude',fontsize=xlabel_size)
    ax1.xaxis.set_tick_params(labelsize=xlabel_size,labelrotation=rot)
    ax1.yaxis.set_tick_params(labelsize=xlabel_size)
    ax1.grid(color = 'k', linestyle = '--', linewidth = grd_w )
    [i.set_linewidth(4) for i in ax1.spines.values()]
    
    
    ax2 = fig.add_subplot(1, 4,2)
    ax2.tricontour(lon2,lat2, slip2, levels=levels, linewidths=0.01, colors='k')
    cntr2 = ax2.tricontourf(lon2,lat2, slip2, levels=levels, cmap=new_cmap) #RdBu_r
    ax2.plot(lon2,lat2, 'ko', ms=ms)
    ax2.set_title('FQ Model 1',fontsize=size_title)
    ax2.set_xlabel('Longitude',fontsize=xlabel_size)
    ax2.xaxis.set_tick_params(labelsize=xlabel_size,labelrotation=rot)
    ax2.yaxis.set_tick_params(labelsize=xlabel_size)
    ax2.grid(color = 'k', linestyle = '--', linewidth = grd_w)
    [i.set_linewidth(4) for i in ax2.spines.values()]

    
    ax3 = fig.add_subplot(1, 4,3)
    ax3.tricontour(lon3,lat3, slip3, levels=levels, linewidths=0.01, colors='k')
    cntr2 = ax3.tricontourf(lon3,lat3, slip3, levels=levels, cmap=new_cmap) #RdBu_r
    ax3.plot(lon3,lat3, 'ko', ms=ms)
    ax3.set_title('FQ Model 2',fontsize=size_title)
    ax3.set_xlabel('Longitude',fontsize=xlabel_size)
    ax3.xaxis.set_tick_params(labelsize=xlabel_size,labelrotation=rot)
    ax3.yaxis.set_tick_params(labelsize=xlabel_size)
    ax3.grid(color = 'k', linestyle = '--', linewidth = grd_w)
    [i.set_linewidth(4) for i in ax3.spines.values()]
    
    
    ax4 = fig.add_subplot(1, 4,4)
    ax4.tricontour(lon4,lat4, slip4, levels=levels, linewidths=0.01, colors='k')
    cntr2 = ax4.tricontourf(lon4,lat4, slip4, levels=levels, cmap=new_cmap) #RdBu_r
    ax4.plot(lon4,lat4, 'ko', ms=ms)
    ax4.set_title('FQ Model 3',fontsize=size_title)
    ax4.set_xlabel('Longitude',fontsize=xlabel_size)
    ax4.xaxis.set_tick_params(labelsize=xlabel_size,labelrotation=rot)
    ax4.yaxis.set_tick_params(labelsize=xlabel_size)
    ax4.grid(color = 'k', linestyle = '--', linewidth = grd_w)
    [i.set_linewidth(4) for i in ax4.spines.values()]
    
    cb = plt.colorbar(cntr2)
    cb.set_label('Slip (m)',fontsize = xlabel_size)
    cb.ax.tick_params(labelsize=xlabel_size)
    
    
    figpath = os.getcwd() +'/fig.mean_n_fqs_srfs_'+ rupt_name +'.png'
    plt.savefig(figpath, bbox_inches='tight', dpi=100)
    
    
#%% plot_srf_ptsouce
def plot_srf_ptsouce(rupt_name,rupt,max_slip=24):
    lon,lat,x,y,slip = extract_rupt_data(rupt,slip_percent =-10)
    lon2,lat2,x2,y2,slip2 = extract_rupt_data(rupt,slip_percent =0)
    
    slip[slip>max_slip]=max_slip-.01
    slip2[slip2>max_slip]=max_slip-.01
    
    fig = plt.figure(figsize=(40,25), dpi=100)
    fig.suptitle('Source Rupture Model and Point Sources ('+rupt_name+')', fontsize=70)
    
    levels = np.linspace(0, max_slip, max_slip+1)
    
    xlabel_size = 57
    ms = 15
    
    ax1 = fig.add_subplot(1, 2,1)
    ax1.tricontour(lon,lat, slip, levels=levels, linewidths=0.01, colors='k')
    cntr2 = ax1.tricontourf(lon,lat, slip, levels=levels, cmap=new_cmap) #RdBu_r
    ax1.plot(lon,lat, 'ko', ms=ms)
    ax1.set_title('FQ Model 3',fontsize=xlabel_size+10)
    ax1.set_xlabel('Longitude',fontsize=xlabel_size)
    ax1.set_ylabel('Latitude',fontsize=xlabel_size)
    ax1.xaxis.set_tick_params(labelsize=xlabel_size,labelrotation=25)
    ax1.yaxis.set_tick_params(labelsize=xlabel_size)
    x_min, x_max = ax1.get_xlim()
    y_min, y_max = ax1.get_ylim()
    ax1.grid(color = 'k', linestyle = '-', linewidth = 0.5)
    
    ax2 = fig.add_subplot(1, 2,2)
    ax2.scatter(lon2,lat2,marker='o',c=slip2,s=200,cmap=new_cmap,vmin=0)#RdBu_r
    ax2.set_ylabel('Latitude',fontsize=xlabel_size)
    ax2.set_xlabel('Longitude',fontsize=xlabel_size)
    ax2.xaxis.set_tick_params(labelsize=xlabel_size,labelrotation=25)
    ax2.yaxis.set_tick_params(labelsize=xlabel_size)
    ax2.set_title('Point Sources (slip > 0)',fontsize=xlabel_size+10)
    ax2.quiver(lon2,lat2,x2,y2,color='green',width=0.002)
    ax2.grid(color = 'k', linestyle = '-', linewidth = 0.5)
    ax2.set_xlim([x_min, x_max])
    ax2.set_ylim([y_min, y_max])
    
    cb = plt.colorbar(cntr2)
    cb.set_label('Slip (m)',fontsize = xlabel_size)

    cb.ax.tick_params(labelsize=xlabel_size)
    
    #figpath = os.getcwd() +'/fig.srf_ptsources_'+ rupt[-30:-5] +'.png'
    figpath = os.getcwd() +'/fig.srf_ptsources_'+ rupt_name +'.png'
    plt.savefig(figpath, bbox_inches='tight', dpi=100)

    

#%% Driver Ibaraki SRCMOD
rupt_name = 'Ibaraki_SRCMOD'
rupt1 = '/Users/oluwaseunfadugba/Documents/Projects/TsE_ValerieDiego/TsE_1D_vs_3D/'+\
            '1D_Modeling_using_FQs_Mudpy/Running_FakeQuakes_now/ibaraki2011_srcmod/ibaraki2011_srcmod_mesh.rupt'

path = '/Users/oluwaseunfadugba/Documents/Projects/TsE_ValerieDiego/TsE_1D_vs_3D/'+\
        '1D_Modeling_using_FQs_Mudpy/Running_FakeQuakes_now/ibaraki2011_srcmod/ruptures/'

rupt2 = path+'ibaraki2011_srcmod.000005.rupt' #1
rupt3 = path+'ibaraki2011_srcmod.000009.rupt' #6
rupt4 = path+'ibaraki2011_srcmod.000006.rupt' #9

max_slip=10 #25
title = 'Mean Rupture Model and Example FQ MOdels (Ibaraki 2011 SRCMOD)'
plot_4srf(rupt_name,rupt1,rupt2,rupt3,rupt4,max_slip,title)

plot_srf_ptsouce(rupt_name,rupt2,max_slip)



#%% Driver Ibaraki ZHENG
rupt_name = 'Ibaraki_ZHENG'

rupt1 = '/Users/oluwaseunfadugba/Documents/Projects/TsE_ValerieDiego/TsE_1D_vs_3D/'+\
            '1D_Modeling_using_FQs_Mudpy/Running_FakeQuakes_now/ibaraki2011_zheng1/ibaraki2011_zheng1_mesh.rupt'

path = '/Users/oluwaseunfadugba/Documents/Projects/TsE_ValerieDiego/TsE_1D_vs_3D/'+\
        '1D_Modeling_using_FQs_Mudpy/Running_FakeQuakes_now/ibaraki2011_zheng1/ruptures/'

rupt2 = path+'ibaraki2011_zheng1.000000.rupt' #1
rupt3 = path+'ibaraki2011_zheng1.000001.rupt' #6
rupt4 = path+'ibaraki2011_zheng1.000011.rupt' #9

max_slip=10 #25
title = 'Ibaraki 2011 (Zheng Rupture Model)'
plot_4srf(rupt_name,rupt1,rupt2,rupt3,rupt4,max_slip,title)

plot_srf_ptsouce(rupt_name,rupt2,max_slip)



#%% Driver Iwate ZHENG
rupt_name = 'Iwate2011_ZHENG'

rupt1 = '/Users/oluwaseunfadugba/Documents/Projects/TsE_ValerieDiego/TsE_1D_vs_3D/'+\
            '1D_Modeling_using_FQs_Mudpy/Running_FakeQuakes_now/iwate2011_zheng1/iwate2011_zheng1_mesh.rupt'

path = '/Users/oluwaseunfadugba/Documents/Projects/TsE_ValerieDiego/TsE_1D_vs_3D/'+\
            '1D_Modeling_using_FQs_Mudpy/Running_FakeQuakes_now/iwate2011_zheng1/ruptures/'

rupt2 = path+'iwate2011_zheng1.000000.rupt' #1
rupt3 = path+'iwate2011_zheng1.000003.rupt' #6
rupt4 = path+'iwate2011_zheng1.000004.rupt' #9

max_slip=10#25
title = 'Iwate 2011 (Zheng Rupture Model)'
plot_4srf(rupt_name,rupt1,rupt2,rupt3,rupt4,max_slip,title)

plot_srf_ptsouce(rupt_name,rupt2,max_slip)


#%% Driver Miyagi ZHENG
rupt_name = 'Miyagi2011_ZHENG'

rupt1 = '/Users/oluwaseunfadugba/Documents/Projects/TsE_ValerieDiego/TsE_1D_vs_3D/'+\
            '1D_Modeling_using_FQs_Mudpy/Running_FakeQuakes_now/miyagi2011a_zheng1/miyagi2011a_zheng1_mesh.rupt'

path = '/Users/oluwaseunfadugba/Documents/Projects/TsE_ValerieDiego/TsE_1D_vs_3D/'+\
            '1D_Modeling_using_FQs_Mudpy/Running_FakeQuakes_now/miyagi2011a_zheng1/ruptures/'

rupt2 = path+'miyagi2011a_zheng1.000000.rupt' #1
rupt3 = path+'miyagi2011a_zheng1.000003.rupt' #6
rupt4 = path+'miyagi2011a_zheng1.000002.rupt' #9

max_slip=10
title = 'Miyagi 2011 (Zheng Rupture Model)'
plot_4srf(rupt_name,rupt1,rupt2,rupt3,rupt4,max_slip,title)

plot_srf_ptsouce(rupt_name,rupt2,max_slip)

#%% Driver Miyagi Hayes
rupt_name = 'Miyagi2011_Hayes'

rupt1 = '/Users/oluwaseunfadugba/Documents/Projects/TsE_ValerieDiego/TsE_1D_vs_3D/'+\
            '1D_Modeling_using_FQs_Mudpy/Running_FakeQuakes_now/miyagi2011a_usgs/miyagi2011a_usgs_mesh.rupt'

path = '/Users/oluwaseunfadugba/Documents/Projects/TsE_ValerieDiego/TsE_1D_vs_3D/'+\
            '1D_Modeling_using_FQs_Mudpy/Running_FakeQuakes_now/miyagi2011a_usgs/ruptures/'

rupt2 = path+'miyagi2011a_usgs.000001.rupt' #1
rupt3 = path+'miyagi2011a_usgs.000002.rupt' #6
rupt4 = path+'miyagi2011a_usgs.000000.rupt' #9

max_slip=10
title = 'Miyagi 2011 (Hayes Rupture Model)'
plot_4srf(rupt_name,rupt1,rupt2,rupt3,rupt4,max_slip,title)

plot_srf_ptsouce(rupt_name,rupt2,max_slip)




#%% Driver Tokachi Hayes
rupt_name = 'tokachi2003_Hayes'

rupt1 = '/Users/oluwaseunfadugba/Documents/Projects/TsE_ValerieDiego/TsE_1D_vs_3D/'+\
            '1D_Modeling_using_FQs_Mudpy/Running_FakeQuakes_now/tokachi2003_usgs/tokachi2003_usgs_mesh.rupt'

path = '/Users/oluwaseunfadugba/Documents/Projects/TsE_ValerieDiego/TsE_1D_vs_3D/'+\
            '1D_Modeling_using_FQs_Mudpy/Running_FakeQuakes_now/tokachi2003_usgs/ruptures/'

rupt2 = path+'tokachi2003_usgs.000000.rupt' #1
rupt3 = path+'tokachi2003_usgs.000001.rupt' #6
rupt4 = path+'tokachi2003_usgs.000004.rupt' #9

max_slip=10
title = 'Tokachi 2003 (Hayes Rupture Model)'
plot_4srf(rupt_name,rupt1,rupt2,rupt3,rupt4,max_slip,title)

plot_srf_ptsouce(rupt_name,rupt2,max_slip)


#%% Driver Tokachi SRCMOD1
rupt_name = 'tokachi2003_srcmod3'

rupt1 = '/Users/oluwaseunfadugba/Documents/Projects/TsE_ValerieDiego/TsE_1D_vs_3D/'+\
            '1D_Modeling_using_FQs_Mudpy/Running_FakeQuakes_now/tokachi2003_srcmod3/tokachi2003_srcmod3_mesh.rupt'

path = '/Users/oluwaseunfadugba/Documents/Projects/TsE_ValerieDiego/TsE_1D_vs_3D/'+\
            '1D_Modeling_using_FQs_Mudpy/Running_FakeQuakes_now/tokachi2003_srcmod3/ruptures/'

rupt2 = path+'tokachi2003_srcmod3.000001.rupt' #1
rupt3 = path+'tokachi2003_srcmod3.000004.rupt' #6
rupt4 = path+'tokachi2003_srcmod3.000000.rupt' #9

max_slip=10#20
title = 'Tokachi 2003 (SRCMOD3 Rupture Model)'
plot_4srf(rupt_name,rupt1,rupt2,rupt3,rupt4,max_slip,title)

plot_srf_ptsouce(rupt_name,rupt2,max_slip)


# ####################################################################
end = time.time()
time_elaps = end - start
if time_elaps < 60:
    print(f'Duration: {round(time_elaps)} seconds')
else:
    print(f'Duration: {round(time_elaps/60)} minutes')  
  

