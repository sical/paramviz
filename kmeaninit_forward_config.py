#!/usr/bin/python

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

import sys
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from KmeansTrace import KmeansTrace
from DataGrid import DataGrid

folder = './' 

def KmeansGrid(Data2D, K, grid):
    
    listeTraceLabels  = []
    listeTraceCenters = []
    listeTraceInertia = []
    listeTraceNbIter  = []
    
    for c in grid.get_grille():

        aKmeans  = KmeansTrace(Data2D, K, c, grid)
        Labels, Centers, Inertia, NbIter = aKmeans.getAll()
        listeTraceLabels.append(Labels)  
        listeTraceCenters.append(Centers) 
        listeTraceInertia.append(Inertia) 
        listeTraceNbIter.append(NbIter) 
        
#        print(aKmeans.getLabelsCentersInertia())
#        print(aKmeans.getCenter_Init())
#        print(aKmeans.getLabels())

    return listeTraceLabels, listeTraceCenters, listeTraceInertia, listeTraceNbIter

if __name__ == "__main__":

    import time
    start_time = time.time()

""" Arrays to store the output of the K-means calculation """
    hgrams=[]
    cargo_labels=[]
    cargo_iterations=[]
    cargo_inertia=[]
    cargo_solutions=[]
    cargo_config=[]
    counter=1
    
""" Arrays to store all possible configurations of Dx, Dy, K """
    gridx=[]
    gridy=[]
    gridk=[]
    gridsize=[]
    
""" Read the input data and count the datapoints to define the grid boundaries """
    filename = './data/UnevenlySizedBlobs.csv' 
    data_points = pd.read_csv(filename, sep='\t', header=None)
    datacount   = len(data_points)
    
    # 
    for i in range(2,datacount):
        dlim = int(datacount/i)
        for j in range(1,dlim+1):
            klim = (i*j)-1
            for l in range(2,klim+1):

                K = l
                nx, ny = i,j
                index = nx*ny
#                gridx.append([nx,ny,K,index])
                gridx.append(nx)
                gridy.append(ny)
                gridk.append(K)
                gridsize.append(index)
  
    hs2=pd.DataFrame({"nx":gridx[:],"ny":gridy[:],"nk":gridk[:],"gridsize":gridsize[:]})
    hs_sorted=hs2.sort(['gridsize'],ascending=True)

#    for i in range (0,len(hs2)):
#        sorted_gridx.append(hs_sorted.nx[i])
#        sorted_gridy.append(hs_sorted.ny[i])
#        sorted_gridk.append(hs_sorted.nk[i])
#        sorted_gridsize.append(hs_sorted.gridsize[i])
#        
#    hs3=pd.DataFrame({"snx":sorted_gridx[:],"sny":sorted_gridy[:],"snk":sorted_gridk[:],"gridsize":sorted_gridsize[:]})
    
    number_of_config=300
    for i in range (0,number_of_config):
    
                K = hs_sorted.nk[i]
                nx, ny = hs_sorted.nx[i],hs_sorted.ny[i]
                #print(nx,ny,K)
                #print(nx*ny)

""" Data reading """
#                #filename = './data/UnequalVar.csv'
#                #filename = './data/Mixture2D.csv'              #----> nbreuses convergences avec K =3 (un peu moins avec K=2)
                filename = './data/UnevenlySizedBlobs.csv'      #----> nbreuses convergences avec K =3 (un seule avec K =2)
                #filename = './data/AnisotropBlob.csv'
                df       = pd.read_csv(filename, sep='\t', header=None)
                Data2D   = df.values
#            
""" All about the digital grid for center init of kmeans """
                grid = DataGrid(Data2D, nx, ny, K)
            
""" Kmeans computation with center init on the grid """
                listeTraceLabels, listeTraceCenters, listeTraceInertia, listeTraceNbIter = KmeansGrid(Data2D, K, grid)
                
                hgrams.append(np.histogram(listeTraceNbIter, density=True, bins=np.max(listeTraceNbIter)))
                
                shortInertia=list(set(listeTraceInertia))
                shorter=np.round(shortInertia,1)
                
                #print('Label of each point \n', listeTraceLabels)
#                print('Nb iter to reach convergence for each init \n', listeTraceNbIter)
#                print('Nb iter to reach convergence for each init \n', listeTraceCenters)
#                print('Unique different value of inertia ', shorter)
#                print('Solutions',len(list(set(listeTraceInertia))))
                #print("--- %s seconds ---" % (time.time() - start_time))
#                print(K,nx,ny)
                
                cargo_iterations.append(listeTraceNbIter)
                cargo_labels.append(listeTraceLabels)
                cargo_inertia.append(shorter)
                cargo_solutions.append(len(list(set(listeTraceInertia))))
                cargo_config.append(str(l)+str(i)+str(j))
                counter=counter+1
                
#np.count_nonzero(A==B)

""" Output JSON file """

hs_diag=pd.DataFrame({"iterations":cargo_iterations[:],"inertia":cargo_inertia[:],"solutions":cargo_solutions[:],"config":cargo_config[:]})
#hs2=hs2.fillna(999)
hs_diag.reset_index().to_json(orient='records',path_or_buf=folder+'Num_config_'+str(number_of_config)+'.json')

print("--- %s seconds ---" % (time.time() - start_time))

""" Python Plot """

fig0 = plt.figure()

""" Distribution of the data """
marquershape = 'o' 
marquersize  = 10 
ax  = fig0.add_subplot(1,1,1)
thex=range(0,len(gridsize))
ax.plot(thex,hs_sorted.gridsize[:], ls='none', marker='o', markersize=2, color=(0.75, 0.75, 0.75))
