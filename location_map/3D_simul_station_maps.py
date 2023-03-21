#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 00:25:18 2023

@author: oluwaseunfadugba
"""

def extract_source_pts(rupt):
    #Read mudpy file
    import numpy as np
    f=np.genfromtxt(rupt)
    
    lon_s = np.array([])
    lat_s = np.array([])
    depth_s = np.array([])
    
    #loop over subfaults
    for kfault in range(len(f)):

        zero_slip=False

        #Get subfault parameters
        lon=f[kfault,1]
        lat=f[kfault,2]
        depth=f[kfault,3]*1000 #in m for sw4
        strike=f[kfault,4]
        dip=f[kfault,5]
        area=f[kfault,10]*f[kfault,11] #in meters, cause this isn't dumb SRF
        #tinit=f[kfault,12]+time_pad
        #rake=rad2deg(arctan2(f[kfault,9],f[kfault,8]))
        slip=np.sqrt(f[kfault,8]**2+f[kfault,9]**2)
        rise_time=f[kfault,7]
        rigidity=f[kfault,13]

        #If subfault has zero rise time or zero slip
        zero_slip=False
        if slip==0:
            zero_slip=True
            #print('Zero slip at '+str(kfault))
        elif rise_time==0:
            slip=0
            zero_slip=True
            #print('Zero rise time at '+str(kfault))     

        #make rake be -180 to 180
        #if rake>180:
        #    rake=rake-360

        if zero_slip==False:
            
            lon_s = np.append(lon_s, lon)
            lat_s = np.append(lat_s, lat)
            depth_s = np.append(depth_s, depth)
           
    return lon_s,lat_s,depth_s

def plot_eq_sta(title_name,infile,homerupt,x_range,y_range):
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np

    print(len(homerupt))
    fig = plt.figure(figsize=(80, 40), facecolor='white',dpi=100)
    
    fig.tight_layout(w_pad=200)
    
    
    fig.suptitle(title_name,fontsize=200)
    ax1 = fig.add_subplot(1, 2,1)
    msize = 40

    # Read in the metadata file
    flatfile_res_dataframe = pd.read_csv(infile)   
    # flatfile_res_dataframe = flatfile_res_dataframe[flatfile_res_dataframe['rupt_no']==rupt_no]
    # flatfile_res_dataframe = flatfile_res_dataframe[flatfile_res_dataframe['SNR_obs']>=3]
    # flatfile_res_dataframe = flatfile_res_dataframe[flatfile_res_dataframe['hypdist']<1000]

    lon = np.array(flatfile_res_dataframe['stlon']) 
    lat = np.array(flatfile_res_dataframe['stlat']) 
    obs_snr = np.array(flatfile_res_dataframe['SNR_obs']) 
    hypdist = np.array(flatfile_res_dataframe['hypdist']) 

    hyplon = np.array(flatfile_res_dataframe['hyplon']) 
    hyplat = np.array(flatfile_res_dataframe['hyplat']) 
    
    #print(hyplon, hyplat)
    scatter = plt.scatter(lon, lat,marker="<",s = msize*40, c= obs_snr,cmap='viridis') #obs_snr c='b',
    
    plt.colorbar(scatter)
    plt.grid()

    y_buf = 1000*40/120000
    x_buf = 1000*40/80000
    xc1 = x_range[0]-x_buf
    xc2 = x_range[1]+x_buf
    yc1 = y_range[0]-y_buf
    yc2 = y_range[1]+y_buf

    # plt.plot([x_range[0], x_range[1], x_range[1],x_range[0], x_range[0]],
    #          [y_range[0], y_range[0], y_range[1], y_range[1], y_range[0]], '--', \
    #               label='All 3DVel Domain', linewidth=7)

    # plt.plot([xc1, xc2, xc2, xc1, xc1],\
    #          [yc1, yc1, yc2, yc2, yc1], '--', \
    #               label='All 3DVel Domain', linewidth=7)

    # plot ruptures
    for i in range(len(homerupt)):
        [lon_s,lat_s,depth_s] = extract_source_pts(homerupt[i]); 
        plt.scatter(lon_s,lat_s,  c='m',s=msize*30) #'m-', marker=(5, 1),
    
    #plt.scatter(hyplon, hyplat,marker='>',c='r',s = msize*70)
    
    ax1.set_xlabel('longitude',fontsize=msize*2.5)
    ax1.set_ylabel('latitude',fontsize=msize*2.5)
    x_dim = (xc2-xc1)*80000/1000
    y_dim = (yc2-yc1)*120000/1000

    exlat = [min(lat), max(lat)]
    exlon = [min(lon), max(lon)]

    ax2 = fig.add_subplot(1, 2,2)
    scatter = plt.scatter(hypdist, obs_snr,s = msize*30)
    ax2.set_xlabel('distance (km)',fontsize=msize*2.5)
    ax2.set_ylabel('Observed SNR',fontsize=msize*2.5)
    plt.grid()

    print(title_name)
    print('================')
    print('no of stations:',len(lon))
    print('')
    print('orig_dim_lat:',round(exlat[0],2),round(exlat[1],2),round((exlat[1]-exlat[0])*120000/1000,1),'km')
    print('orig_dim_lon:',round(exlon[0],2),round(exlon[1],2),round((exlon[1]-exlon[0])*80000/1000,1),'km')
    print('')
    print('corners = [',round(xc1,2),',',round(yc1,2),']')
    print('x_dim = ',round(x_dim,1),'km')
    print('y_dim = ',round(y_dim,1),'km')

    plt.rcParams['xtick.labelsize']=msize*2.5 #xtick.labelsize
    plt.rcParams['ytick.labelsize']=msize*2.5
    plt.savefig('fig.'+title_name+'.png', bbox_inches='tight', dpi=100)
    
    
    
#%% Ibaraki 2011
title_name='Ibaraki_2011'
infile = '/Users/oluwaseunfadugba/Documents/Projects/TsE_ValerieDiego/TsE_1D_vs_3D/'+\
    '3D_Modeling_using_SW4/Running_Simulations/4_Model_results/All_05Hz_Results/'+\
        'flatfile_ibaraki2011_srcmod_srf3d_talapas_0.49Hz_Residuals.csv'

rupt1 = '/Users/oluwaseunfadugba/Documents/Projects/TsE_ValerieDiego/TsE_1D_vs_3D/'+\
    '1D_Modeling_using_FQs_Mudpy/Running_FakeQuakes_now/ibaraki2011_srcmod/ruptures/'+\
        'ibaraki2011_srcmod.000005.rupt'
#rupt2 = '/Users/oluwaseunfadugba/Documents/Projects/TsE_ValerieDiego/TsE_1D_vs_3D/'+\
#    '1D_Modeling_using_FQs_Mudpy/Running_FakeQuakes_now/ibaraki2011_zheng1/ruptures/'+\
#        'ibaraki2011_zheng1.000001.rupt'

x_range = [130.5,143.3]
y_range = [32.2,39]

plot_eq_sta(title_name,infile,[rupt1],x_range,y_range) #,rupt2

# I will use lat0,lon0 = 31.87,130
# x,y_dim = 1100km, 900km    



#%% Miyagi 2011
title_name='Miyagi_2011'
infile = '/Users/oluwaseunfadugba/Documents/Projects/TsE_ValerieDiego/TsE_1D_vs_3D/'+\
    '3D_Modeling_using_SW4/Running_Simulations/4_Model_results/All_05Hz_Results/'+\
        'flatfile_miyagi2011a_usgs_srf3d_talapas_0.49Hz_Residuals.csv'
#rupt1 = '/Users/oluwaseunfadugba/Documents/Projects/TsE_ValerieDiego/TsE_1D_vs_3D/1D_Modeling_using_FQs_Mudpy/Running_FakeQuakes_now/miyagi2011a_usgs/ruptures/miyagi2011a_usgs.000001.rupt'
rupt2 = '/Users/oluwaseunfadugba/Documents/Projects/TsE_ValerieDiego/TsE_1D_vs_3D/'+\
    '1D_Modeling_using_FQs_Mudpy/Running_FakeQuakes_now/miyagi2011a_zheng1/'+\
        'ruptures/miyagi2011a_zheng1.000000.rupt'

x_range = [134.2,145]
y_range = [33.5,44.4]

plot_eq_sta(title_name,infile,[rupt2],x_range,y_range)#,rupt1

# I will use lat0,lon0 = 33.17,133.7
# x,y_dim = 950km, 1400km

#%% Iwate 2011
title_name='Iwate_2011'
infile = '/Users/oluwaseunfadugba/Documents/Projects/TsE_ValerieDiego/TsE_1D_vs_3D/'+\
    '3D_Modeling_using_SW4/Running_Simulations/4_Model_results/All_05Hz_Results/'+\
        'flatfile_iwate2011_zheng1_srf3d_talapas_0.49Hz_Residuals.csv'
rupt1 = '/Users/oluwaseunfadugba/Documents/Projects/TsE_ValerieDiego/TsE_1D_vs_3D/'+\
    '1D_Modeling_using_FQs_Mudpy/Running_FakeQuakes_now/iwate2011_zheng1/ruptures/'+\
        'iwate2011_zheng1.000000.rupt'

x_range = [134,145]
y_range = [33.5,45.3]

plot_eq_sta(title_name,infile,[rupt1],x_range,y_range)

# I will use lat0,lon0 = 33.17,133.5
# x,y_dim = 960km, 1500km


#%% Tokachi 2003
title_name='Tokachi_2003'
infile = '/Users/oluwaseunfadugba/Documents/Projects/TsE_ValerieDiego/TsE_1D_vs_3D/'+\
    '3D_Modeling_using_SW4/Running_Simulations/4_Model_results/All_05Hz_Results/'+\
        'flatfile_tokachi2003_srcmod3_srf3d_talapas_0.49Hz_Residuals.csv'
rupt1 = '/Users/oluwaseunfadugba/Documents/Projects/TsE_ValerieDiego/TsE_1D_vs_3D/'+\
    '1D_Modeling_using_FQs_Mudpy/Running_FakeQuakes_now/tokachi2003_usgs/ruptures/'+\
        'tokachi2003_usgs.000000.rupt'
#rupt2 = '/Users/oluwaseunfadugba/Documents/Projects/TsE_ValerieDiego/TsE_1D_vs_3D/1D_Modeling_using_FQs_Mudpy/Running_FakeQuakes_now/tokachi2003_srcmod3/ruptures/tokachi2003_srcmod3.000001.rupt'

x_range = [138,147]
y_range = [36.4,45.8]

plot_eq_sta(title_name,infile,[rupt1],x_range,y_range)#,rupt2

# I will use lat0,lon0 = 33.07,137.5
# x,y_dim = 800km, 1200km




#%% create figure
import matplotlib.pyplot as plt
import os
from string import ascii_lowercase as alphab
from matplotlib.cbook import get_sample_data

fig = plt.figure(figsize=(20, 20))

# setting values to rows and column variables
rows = 3
columns = 2


#%% Figure S5
comp_all = ['Ibaraki_2011','Miyagi_2011','Iwate_2011','Tokachi_2003','3D_domain']

for i in range(5):
    # reading images
    Image1 = plt.imread(os.getcwd()+"/fig."+comp_all[i]+".png")
    
    fig.add_subplot(rows, columns, i+1)
    plt.imshow(Image1)
    plt.axis('off')
    plt.text(0.5, 0.5, '('+alphab[i].upper()+')', color='k', fontsize=25)#, weight='bold')
    plt.tight_layout()
      
# # Adding mean difference residual cartoon figure
# im = plt.imread(get_sample_data(os.getcwd()+'/fig.3D_domain.png'))
# newax = fig.add_axes([0, -0.3, 0.4, 0.4], anchor='NE', zorder=-1)
# newax.imshow(im)
# newax.axis('off')

fig.savefig('figS5.3D_sw4_0.5Hz_setup.png',dpi=500,\
    bbox_inches='tight',facecolor='white', edgecolor='none')
