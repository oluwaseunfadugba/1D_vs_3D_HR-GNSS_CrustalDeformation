#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 15 09:39:46 2021

@author: dmelgarm
"""
from numpy import genfromtxt,where,savetxt
import sys

sys.path.append('/Users/oluwaseunfadugba/Documents/Projects/TsE_ValerieDiego/TsE_1D_vs_3D/Japan_Trench_Mesh_GMSH/gmsh/')

import gmsh_tools

mshout_tmp = '/Users/oluwaseunfadugba/Documents/Projects/TsE_ValerieDiego/TsE_1D_vs_3D/Japan_Trench_Mesh_GMSH/Japan_trench_tmp.mshout'
mshout = '/Users/oluwaseunfadugba/Documents/Projects/TsE_ValerieDiego/TsE_1D_vs_3D/Japan_Trench_Mesh_GMSH/Japan_trench.mshout'
mshin = '/Users/oluwaseunfadugba/Documents/Projects/TsE_ValerieDiego/TsE_1D_vs_3D/Japan_Trench_Mesh_GMSH/Japan.msh'

mudpy_fault = '/Users/oluwaseunfadugba/Documents/Projects/TsE_ValerieDiego/TsE_1D_vs_3D/Japan_Trench_Mesh_GMSH/Japan_trench.fault'

gmsh_tools.gmsh2ascii(mshout_tmp,mshin,utm_zone='54',flip_lon=True,flip_strike=True)

mesh=genfromtxt(mshout_tmp)


# I=where(mesh[:,-2]>500)[0]
# mesh[i,-2] -= 180
# i=where(mesh[:,-2]>400)[0]
# mesh[i,-2] -= 180


# i=where((mesh[:,1]>134.6) & (mesh[:,1]<136.7) & (mesh[:,-2]<130))[0]
# mesh[i,-2] += 180

mesh[:,3]*=-1
mesh[:,6]*=-1
mesh[:,9]*=-1
mesh[:,12]*=-1


h=' fault No. , centroid(lon,lat,z[km]) , node2(lon,lat,z[km]) , node3(lon,lat,z[km]) , mean vertex length(km) , area(km^2) , strike(deg) , dip(deg)'
savetxt(mshout,mesh,fmt='%d\t'+12*'%.6f\t'+4*'%.2f\t',header=h)

gmsh_tools.make_mudpy_fault(mshout,mudpy_fault)