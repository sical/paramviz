#!/usr/bin/python

""" Kmeans algorithm and output diagnostics in JSON """
""" From http://flothesof.github.io/k-means-numpy.html """

# Load modules
import numpy as np
import pandas as pd

""" Choose the dataset and the output directory """
folder = './' 
target = '/Users/jmartinez/Sites/kmeansvisu_cargo/Data/'
#dataset = 'UnequalVar'
#dataset = 'Mixture2D' 
dataset = 'AnisotropBlob'

""" Read the data and calculate length """
filename = './data/'+dataset+'.csv'
data_points = pd.read_csv(filename, sep='\t', header=None)
datacount   = len(data_points)
dataplot = data_points.values

""" Modules for the Kmeans calculation """
def initialize_centroids(points, clusters):
        """returns k centroids from the initial points"""
        centroids = points.copy()
        np.random.shuffle(centroids)
        return centroids[:clusters]

def closest_centroid(points, centroids):
        """returns an array containing the index to the nearest centroid for each point"""
#        distances = np.sqrt(((points - centroids[:, np.newaxis])**2).sum(axis=2))
        distances = np.abs(points - centroids[:, np.newaxis]).sum(axis=2)
        return np.argmin(distances, axis=0)
    
def calculate_inertia(points, closest, centroids):
#        distances = np.sqrt(((points - centroids[:, np.newaxis])**2).sum(axis=2))
        distances = np.abs(points - centroids[:, np.newaxis]).sum(axis=2)
        return distances

def move_centroids(points, closest, centroids):
        """returns the new centroids assigned from the points closest to them"""
        return np.array([points[closest==k].mean(axis=0) for k in range(centroids.shape[0])])

""" Parameter definition """
samples=100 
iterations=40
clusters=3

""" Storage of the variables """
cargo_centroid=[]
cargo_inertia=[]
supercargo_inertia=[]
cargo_clustersize=[]
cargo_shifts=[]
supercargo_shifts=[]
phase_x=[]
phase_y=[]
sphase_x=[]
sphase_y=[]

""" Loop for many initializations, or samples, over Kmeans """
for s in range(0,samples):
    """ Initialize the random centroids """
    c = initialize_centroids(dataplot,clusters)
    for i in range(0,iterations):
        """ Move the centroids """
        mov = move_centroids(dataplot, closest_centroid(dataplot,c), c)
        c = mov
        cargo_centroid.append(c)
        
        """ Calculate the sum of distances within each cluster """
        inertia=calculate_inertia(dataplot, closest_centroid(dataplot,c), c)
        for n in range(0,len(inertia)):
            cargo_inertia.append(np.sum(inertia[n]))
        supercargo_inertia.append(cargo_inertia)
        cargo_inertia=[]
        
        """ Count the number of datapoints in each cluster """
        groups=closest_centroid(dataplot,c)
        for m in range(0,clusters):
           cargo_shifts.append(np.sum(np.count_nonzero(groups==m))) 
        supercargo_shifts.append(cargo_shifts)
        cargo_shifts=[]
        
    """ Pairing the values for the phase space """
    
    for v in range(0,clusters):
     for j in range(0,iterations):
          phase_x.append(supercargo_shifts[j][v])
          phase_y.append(supercargo_inertia[j][v])
     sphase_x.append(phase_x)
     sphase_y.append(phase_y)    
     phase_x=[]
     phase_y=[]  
     
     """ Save output in JSON format via Pandas """
     phase_pd=pd.DataFrame({"phase_x":sphase_x[v],"phase_y":sphase_y[v]})
     phase_pd.reset_index().to_json(orient='records',path_or_buf=target+dataset+'_PhaseSpace_'+str(s)+'_'+str(v)+'.json')

    sphase_x=[]
    sphase_y=[]
    supercargo_shifts=[]
    supercargo_inertia=[]

    
    
    
    
    
    
    
