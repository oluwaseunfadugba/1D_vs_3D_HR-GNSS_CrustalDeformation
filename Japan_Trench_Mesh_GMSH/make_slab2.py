from numpy import genfromtxt,where,float64,r_,expand_dims,c_,zeros
from glob import glob
import sys

sys.path.append('/Users/oluwaseunfadugba/Documents/Projects/TsE_ValerieDiego/TsE_1D_vs_3D/Japan_Trench_Mesh_GMSH/gmsh/')
from gmsh_tools import xyz2gmsh

#Gmsh output file name
gmsh_out='/Users/oluwaseunfadugba/Documents/Projects/TsE_ValerieDiego/TsE_1D_vs_3D/Japan_Trench_Mesh_GMSH/Japan.gmsh'

#Depth filter
maxdepth=80
#Line filters
#north line
L1x1 = 148.03 ; L1y1 = 40.95
L1x2 = 141.97 ; L1y2 = 45.54
#south line
L2x1 = 144.86 ; L2y1 = 33.04
L2x2 = 137.49 ; L2y2 = 35.88

#Contours files 
contour_files= ['/Users/oluwaseunfadugba/Documents/Projects/TsE_ValerieDiego/TsE_1D_vs_3D/Japan_Trench_Mesh_GMSH/japan_trench.contours']


ks=0
for k in range(len(contour_files)):
    f=open(contour_files[k])
    while True:
        line=f.readline().rstrip()
        if not line: #Exit loop
            break
        #Assign line info to array
        if '>' not in line: #It's not a segment header
            if ks==0:
                contours=expand_dims(float64(line.split('\t')),0)
                # r_[contours,expand_dims(float64(line.split('\t')),0)]   
                ks+=1
            else:
                contours=r_[contours,expand_dims(float64(line.split('\t')),0)]   
    f.close()

contours[:,2] *= -1

#Now filter things
#By depth
i=where(contours[:,2]<=maxdepth)[0]
contours=contours[i,:]



#By regions
# Equations of the line filters
m1=(L1y1-L1y2)/(L1x1-L1x2)
b1=L1y1-m1*L1x1
m2=(L2y1-L2y2)/(L2x1-L2x2)
b2=L2y1-m2*L2x1
#Get above line 1
ytest=m1*contours[:,0]+b1
i=where((ytest-contours[:,1])>=0)[0] 
contours=contours[i,:]
#Get below line 2
ytest=m2*contours[:,0]+b2
i=where((ytest-contours[:,1])<=0)[0] 
contours=contours[i,:]


##Write gmsh file
xyz2gmsh(gmsh_out,contours[:,0],contours[:,1],contours[:,2],coord_type='UTM',projection_zone='54')


    
    




