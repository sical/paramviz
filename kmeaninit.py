#!/usr/bin/python

import sys
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from KmeansTrace import KmeansTrace
from DataGrid import DataGrid

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

        #print(aKmeans.getLabelsCentersInertia())
        #print(aKmeans.getCenter_Init())

    return listeTraceLabels, listeTraceCenters, listeTraceInertia, listeTraceNbIter


if __name__ == "__main__":

    import time
    start_time = time.time()

    K = 3
    nx, ny = 5,5

    # Data reading
    #filename = './data/UnequalVar.csv'
    #filename = './data/Mixture2D.csv'              #----> nbreuses convergences avec K =3 (un peu moins avec K=2)
    filename = './data/UnevenlySizedBlobs.csv'      #----> nbreuses convergences avec K =3 (un seule avec K =2)
    #filename = './data/AnisotropBlob.csv'
    df       = pd.read_csv(filename, sep='\t', header=None)
    Data2D   = df.values

    # plots
    marquershape = 'o'
    marquersize  = 10 
    fig = plt.figure()
    ax  = fig.add_subplot(1,1,1)
    ax.plot(Data2D[:,0], Data2D[:, 1], ls='none', marker='o', markersize=2, color=(0.75, 0.75, 0.75))

    # All about the digital grid for center init of kmeans
    grid = DataGrid(Data2D, nx, ny, K)

    # Kmeans computation with center init on the grid
    listeTraceLabels, listeTraceCenters, listeTraceInertia, listeTraceNbIter = KmeansGrid(Data2D, K, grid)
    
    
    print('Nb iter to reach convergence for each init \n', listeTraceNbIter)
    print('Unique different value of inertia ', list(set(listeTraceInertia)))
    print("--- %s seconds ---" % (time.time() - start_time))
