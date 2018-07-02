#!/usr/bin/python

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

import sys
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from KmeansTrace import KmeansTrace
from KmeansTrace_standalone import KmeansTrace_standalone
from DataGrid import DataGrid

def KmeansGrid(Data2D, K, grid):
    
    listeTraceLabels  = []
    listeTraceCenters = []
    listeTraceInertia = []
    listeTraceNbIter  = []

    for c in grid.get_grille():

        aKmeans  = KmeansTrace_standalone(Data2D, K, c, grid)
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

    # Arrays to store the output of the K-means calculation
    hgrams=[]
    cargo_labels=[]
    cargo_iterations=[]
    cargo_inertia=[]
    cargo_solutions=[]
    cargo_config=[]
    counter=1
    
    # Arrays to store all possible configurations of Dx, Dy, K
    gridx=[]
    gridy=[]
    gridk=[]
    gridsize=[]
    
    # Arrays to store the configurations in ascending order by grid size
    sorted_gridx=[]
    sorted_gridy=[]
    sorted_gridk=[]
    sorted_gridsize=[]
    
    # Read the input data and count the datapoints to define the grid boundaries
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
    
    number_of_config=10
    for i in range (0,number_of_config):
    
                K = hs_sorted.nk[i]
                nx, ny = hs_sorted.nx[i],hs_sorted.ny[i]
                #print(nx,ny,K)
                #print(nx*ny)
            
#                # Data reading
#                #filename = './data/UnequalVar.csv'
#                #filename = './data/Mixture2D.csv'              #----> nbreuses convergences avec K =3 (un peu moins avec K=2)
                filename = './data/UnevenlySizedBlobs.csv'      #----> nbreuses convergences avec K =3 (un seule avec K =2)
                #filename = './data/AnisotropBlob.csv'
                df       = pd.read_csv(filename, sep='\t', header=None)
                Data2D   = df.values
#            
                # All about the digital grid for center init of kmeans
                grid = DataGrid(Data2D, nx, ny, K)
            
                # Kmeans computation with center init on the grid
                listeTraceLabels, listeTraceCenters, listeTraceInertia, listeTraceNbIter = KmeansGrid(Data2D, K, grid)
                
                hgrams.append(np.histogram(listeTraceNbIter, density=True, bins=np.max(listeTraceNbIter)))
                
                shortInertia=list(set(listeTraceInertia))
                shorter=np.round(shortInertia,1)
                
                print('Label of each point \n', listeTraceLabels)
#                print('Nb iter to reach convergence for each init \n', listeTraceNbIter)
#                print('Nb iter to reach convergence for each init \n', listeTraceCenters)
#                print('Unique different value of inertia ', shorter)
#                print('Solutions',len(list(set(listeTraceInertia))))
                print("--- %s seconds ---" % (time.time() - start_time))
#                print(K,nx,ny)
                
                cargo_iterations.append(listeTraceNbIter)
                cargo_labels.append(listeTraceLabels)
                cargo_inertia.append(shorter)
                cargo_solutions.append(len(list(set(listeTraceInertia))))
                cargo_config.append(str(l)+str(i)+str(j))
                counter=counter+1
                
#np.count_nonzero(A==B)

#hs2=hs2.fillna(999)
#hs2.reset_index().to_json(orient='records',path_or_buf=folder+'era_wind.json')

""" Python Plot """
#
fig0 = plt.figure()
#fig1 = plt.figure()
#fig2 = plt.figure()
#fig3 = plt.figure()
#fig4 = plt.figure()
#
#                # Distribution of the data
marquershape = 'o'
marquersize  = 10 
ax  = fig0.add_subplot(1,1,1)
thex=range(0,len(gridsize))
ax.plot(thex,hs_sorted.gridsize[:], ls='none', marker='o', markersize=2, color=(0.75, 0.75, 0.75))
#
#                # Total number of solutions for each configuration
#ax  = fig1.add_subplot(1,1,1)
#ax.set_xticklabels(cargo_config)
#ax.plot(range(0,len(cargo_iterations)), cargo_solutions, ls='none', marker='o', markersize=2, color=(0.2, 0.2, 0.2))
#ax.set_xticks(np.arange(0, len(cargo_iterations), 1.0))
#
#                # Solutions for each configuration
#bx  = fig2.add_subplot(1,1,1)
#for xe, ye in zip(range(1,len(cargo_iterations)+1), cargo_inertia[:]):
#    bx.scatter([xe] * len(ye), ye)
#bx.set_xticklabels(cargo_config)
#bx.set_xticks(np.arange(1, len(cargo_iterations)+1, 1.0))
#
#               # Histograms of the number of iterations needed to converge 
#plt.rcParams.update({'font.size': 8})
#m=1
#for k in cargo_iterations:
#
#    cx=fig3.add_subplot(18,1,m)
#    hist, bins = np.histogram(k, bins=np.max(k), range=[1, 30])
#    width = 0.7 * (bins[1] - bins[0])
#    center = (bins[:-1] + bins[1:]) / 2
#    cx.bar(center, hist, align='center', width=1)
#    cx.set_ylabel(cargo_config[m-1],rotation='horizontal', labelpad=20)
#    m=m+1
#
#               # Mean and standard deviation of the number of iterations
#
#means_ite=[]
#for i in cargo_iterations:
#    means_ite.append(np.mean(i))
#    
#stdev_ite=[]
#for i in cargo_iterations:
#    stdev_ite.append(np.std(i))
#
#dx  = fig4.add_subplot(1,1,1)
#dx.errorbar(range(0,len(cargo_iterations)),means_ite,stdev_ite)
#dx.set_xticklabels(cargo_config)
#dx.set_xticks(np.arange(0, len(cargo_iterations), 1.0))