#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 12:51:57 2023

@author: oluwaseunfadugba
"""

import numpy as np
import pygmt
import pandas as pd
import os
import time
from matplotlib import pyplot as plt
import matplotlib
from matplotlib.colors import LinearSegmentedColormap
from glob import glob
import imageio
import moviepy.editor as mp
import cv2

s = 'Slip (m)'

cwd = '/Users/oluwaseunfadugba/Documents/Projects/TsE_ValerieDiego/TsE_1D_vs_3D/'+\
    '1D_vs_3D_HR-GNSS_CrustalDeformation/rupture_figs/'
os.chdir(cwd)
start = time.time()

cdict = {'red':  ((0., 1, 1), (0.03, 1, 1), (0.20, 0, 0), (0.66, 1, 1), (0.89, 1, 1), (1, 0.5, 0.5)),
         'green':((0., 1, 1), (0.03, 1, 1), (0.20, 0, 0), (0.375, 1,1), (0.64, 1, 1), (0.91, 0, 0), (1, 0, 0)),
         'blue': ((0., 1, 1), (0.08, 1, 1), (0.20, 1, 1), (0.34, 1, 1), (0.65, 0, 0), (1, 0, 0))}
whitejet = matplotlib.colors.LinearSegmentedColormap('whitejet',cdict,256)
    
# get colormap and create a colormap object
ncolors = 256
color_array = plt.get_cmap(whitejet)(range(ncolors))
color_array[:,-1] = np.linspace(0.0,1.0,ncolors) # change alpha values
map_object = LinearSegmentedColormap.from_list(name='whitejet_alpha',colors=color_array)
plt.register_cmap(cmap=map_object) # register this new colormap with matplotlib


fig = pygmt.Figure()
#region = [140.9,142.5,35.2,36.5]
region = [132,146,32,45]

with pygmt.clib.Session() as session:
    session.call_module('gmtset', 'FONT 15.3p')
    session.call_module('gmtset', 'MAP_FRAME_TYPE fancy')
    session.call_module('gmtset', 'MAP_FRAME_WIDTH 0.25')

fig.coast(region=region,projection="M15c",land="gray",water="lightblue",borders="1/0.5p",
    shorelines="1/0.5p,black",frame="ag")

pygmt.makecpt(cmap="viridis", series=[0,12])

fig.basemap(frame=["a", '+t"."'])
slipfile = '/Users/oluwaseunfadugba/Documents/Projects/TsE_ValerieDiego/TsE_1D_vs_3D/1D_Modeling_using_FQs_Mudpy/Running_FakeQuakes_now/ibaraki2011_srcmod/ruptures/ibaraki2011_srcmod.000005.gmt'
fig.plot(data = slipfile, color='+z',cmap = True)#, color='+z',cmap = map_object) 

slipfile = '/Users/oluwaseunfadugba/Documents/Projects/TsE_ValerieDiego/TsE_1D_vs_3D/1D_Modeling_using_FQs_Mudpy/Running_FakeQuakes_now/tokachi2003_usgs/ruptures/tokachi2003_usgs.000000.gmt'
fig.plot(data = slipfile, color='+z',cmap = True)#, color='+z',cmap = map_object) 
    
fig.colorbar(frame='af+l"'+s+'"')

fig.show()
fig.savefig('test_plot_rupts_pygmt.png',dpi=2000)