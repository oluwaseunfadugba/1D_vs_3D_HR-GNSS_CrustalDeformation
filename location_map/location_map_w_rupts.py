#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 16 13:04:37 2022

@author: oluwaseunfadugba
"""

import pygmt, os
import pandas as pd
from shapely.geometry import Point, Polygon
import numpy as np
import time
start = time.time()

'''
This code plots the location map showing the GNSS locations and earthquake locations.

To use this code, change the environment to pygmt by running 
"conda activate pygmt" on a new terminal and restart Jupyter notebook.
# conda activate /Users/oluwaseunfadugba/mambaforge/envs/pygmt
'''
current_dir = '/Users/oluwaseunfadugba/Documents/Projects/TsE_ValerieDiego/TsE_1D_vs_3D/'+\
    '1D_vs_3D_HR-GNSS_CrustalDeformation/location_map'

os.chdir(current_dir)

# Set the region for the plot to be slightly larger than the data bounds.
region = [126,150.5,27.5,48]  # entire region
grid_E = [34,47,134.1125,147] # [lat1,lat2,lon1,lon2]
grid_W = [30,37.9,129,141.1]  # [lat1,lat2,lon1,lon2]

# Specify GNSS location file
gflist_filename = '/Users/oluwaseunfadugba/Documents/Projects/TsE_ValerieDiego/'+\
    'TsE_1D_vs_3D/1D_Modeling_using_FQs_Mudpy/Creating_Inputs/GNSS_locations/'+\
        'gflists/ALL_events.GNSS_locs.txt'          

# Earthquake rupture 
homerupt = '/Users/oluwaseunfadugba/Documents/Projects/TsE_ValerieDiego/TsE_1D_vs_3D/'+\
    '1D_Modeling_using_FQs_Mudpy/Running_FakeQuakes_now/'
rupt_no =["5","0","0","0"]
simul = ["ibaraki2011_srcmod","iwate2011_zheng1","miyagi2011a_zheng1","tokachi2003_usgs"]


#%% functions
def extract_source_pts(rupt):
    #Read mudpy file
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

def cs(x1,x2,x3,x4,x5):
    num = 7
    return np.concatenate((np.linspace(x1,x2,num), 
                          np.linspace(x2,x3,num),
                          np.linspace(x3,x4,num),
                          np.linspace(x4,x5,num)))

#%% Extracting GNSS locations

# Read in the metadata file
stadata = pd.read_csv(gflist_filename, sep='\t', header=0,\
                      names=['st_name', 'lat', 'lon', 'elev','sr','gain','unit'])

# Create a Polygon with the coordinates of the East and West 3D velocity model domains
coords_E = [(grid_E[2], grid_E[0]), (grid_E[3], grid_E[0]), 
            (grid_E[3], grid_E[1]), (grid_E[2], grid_E[1])]
coords_W = [(grid_W[2], grid_W[0]), (grid_W[3], grid_W[0]), 
            (grid_W[3], grid_W[1]), (grid_W[2], grid_W[1])]

poly_E = Polygon(coords_E)
poly_W = Polygon(coords_W)

# Initialize array
lon = []; lat = []; elev = []
_lon = []; _lat = []; _elev = []

#print(stadata)
for i in range(len(stadata.index)-1):
    
    lat_pt = float(stadata['lat'][i+1])
    lon_pt = float(stadata['lon'][i+1])
    elev_pt = float(stadata['elev'][i+1])
    
    # Check if the station is within the polygons using the within function
    p1 = Point(lon_pt, lat_pt)
    
    if p1.within(poly_E) == True or p1.within(poly_W) == True:
        lon.append(lon_pt)
        lat.append(lat_pt)
        elev.append(elev_pt)

    else:        
        _lon.append(lon_pt)
        _lat.append(lat_pt)
        _elev.append(elev_pt)
        
print(len(stadata['lon']))
print(len(lon))
print(len(_lon))


#%% PYGMT Figure (Slant rfile)
fig = pygmt.Figure()
 
with pygmt.clib.Session() as session:
    session.call_module('gmtset', 'FONT 15.3p')
    session.call_module('gmtset', 'MAP_FRAME_TYPE fancy')
    session.call_module('gmtset', 'MAP_FRAME_WIDTH 0.25')
    
grid = pygmt.datasets.load_earth_relief(resolution="15s", region=region)
 
fig.coast(region=region,projection="M15c",land="gray",water="lightblue",borders="1/0.5p",
    shorelines="1/0.5p,black",frame="ag")

fig.grdimage(grid=grid, projection="M15c", frame="ag",cmap="geo")
fig.basemap(frame=["a", '+t"."'])

fig.colorbar(frame=["x+lElevation", "y+lkm"],scale=0.001) 

# Plotting the west and east Japan 3D velocity boundaries
east_x= [];  east_x.extend(np.linspace(134.1125,147,4)); 

fig.plot(x=cs(134.1125, 147, 147, 134.1125, 134.1125),y=cs(34, 34, 47, 47, 34),label='East_3DVel_Domain',pen="3p,cyan,-.")
fig.plot(x=cs(129, 141.1, 141.1, 129, 129),y=cs(30, 30, 37.9, 37.9, 30),label='West_3DVel_Domain',pen="3p,green,-")
fig.plot(x=cs(129, 147, 147, 129, 129),y=cs(30, 30, 47, 47, 30),label='All_3DVelPDomain',pen= "3p,blue")

# Plotting profile lines
# fig.plot(x=cs(129, 147, 147, 129, 129),y=cs(32, 32, 32, 32, 32),label='Profile_CD',pen= "3p,magenta")
# fig.plot(x=cs(130, 130, 130, 130, 130),y=cs(30, 30, 47, 47, 30),label='Profile_AB',pen= "3p,magenta")
fig.text(x=129,y=29.3,  text='A', font="16p,Helvetica-Bold,black")
fig.text(x=129,y=47.5,  text='B', font="16p,Helvetica-Bold,black")

fig.text(x=128.3,y=30,  text='C', font="16p,Helvetica-Bold,black")
fig.text(x=147.7,y=30,  text='D', font="16p,Helvetica-Bold,white")

# plot ruptures
pygmt.makecpt(cmap="viridis", series=[0,25])

for i in range(len(simul)):
    slipfile = homerupt+simul[i] +'/ruptures/'+ simul[i]+'.00000'+rupt_no[i]+'.gmt'    
    fig.plot(data = slipfile, color='+z',cmap = True)

# pygmt.config(MAP_TICK_PEN="white",MAP_DEFAULT_PEN="white",MAP_GRID_PEN="white",MAP_FRAME_PEN="white",
#              MAP_TICK_PEN_SECONDARY="white",MAP_GRID_PEN_PRIMARY="white",MAP_GRID_PEN_SECONDARY="white",
#              MAP_TICK_PEN_PRIMARY="white")

fig.colorbar(cmap="viridis", position="g134./30.+w5c/0.3c+h", box=True,
    frame=["x+lSlip(m)"], scale=25)

# Plotting the earthquake locations
fig.plot(x=141.2653,y=36.1083,style="a0.7", color="red",pen="1p,black") ; 
fig.text(x=141.2653,y=36.1083, text='                            Ibaraki 2011', font="16p,Helvetica-Bold,white")
fig.plot(x=142.7815,y=39.8390,style="a0.7", color="red",pen="1p,black"); 
fig.text(x=142.7815,y=39.8390,  text='                            Iwate 2011', font="16p,Helvetica-Bold,white")
fig.plot(x=143.2798,y=38.3285,style="a0.7", color="red",pen="1p,black"); 
fig.text(x=143.2798,y=38.6,  text='                              Miyagi 2011A', font="16p,Helvetica-Bold,white")
fig.plot(x=143.9040,y=41.7750,style="a0.7", color="red",pen="1p,black"); 
fig.text(x=143.9040,y=41.7750,  text='                            Tokachi 2003', font="16p,Helvetica-Bold,white")

fig.plot(x=lon,y=lat,style="t0.3",color="blue",pen="0.9p,black",transparency=50,label='GNSS_Stations')

fig.show()
fig.savefig(current_dir+"/fig1.map_gnssloc_eqrupt.png")

# ####################################################################
end = time.time()
time_elaps = end - start
if time_elaps < 60:
    print(f'Duration: {round(time_elaps)} seconds')
else:
    print(f'Duration: {round(time_elaps/60)} minutes')