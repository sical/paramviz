#!/usr/bin/python

""" Kmeans algorithm and output diagnostics in JSON  """
""" From http://flothesof.github.io/k-means-numpy.html """

""" Load modules """
import numpy as np
import pandas as pd
import seaborn as sns; sns.set(color_codes=True)
import pylab as pl
from scipy import stats
import csv

""" Choose the dataset and the output directory """
folder = './' 
target = '/Users/jmartinez/Sites/kmeansvisu_cargo/distill/Data/ContourPro/'
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
samples=1000 
iterations=8
clusters=3

""" Storage of the variables """
cargo_centroid=[]
supercargo_centroid=[]
cargo_inertia=[]
supercargo_inertia=[]
cargo_clustersize=[]
cargo_shifts=[]
supercargo_shifts=[]
phase_x=[]
phase_y=[]
sphase_x=[]
sphase_y=[]
seacluster_x=[]
seacluster_y=[]


""" Loop for many initializations, or samples, over Kmeans """
for s in range(0,samples):
    """ Initialize the random centroids """
    c = initialize_centroids(dataplot,clusters)
    for i in range(0,iterations):
        """ Move the centroids """
        mov = move_centroids(dataplot, closest_centroid(dataplot,c), c)
        c = mov
        cargo_centroid.append(c)
        supercargo_centroid.append(c[:3])
        cargo_centroid=[]
        seacluster_x.append(c[0][0])
        seacluster_y.append(c[0][1])
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
        
    # Pairing the values for the phase space """
    
#    for v in range(0,clusters):
#     for j in range(0,iterations):
#          phase_x.append(supercargo_shifts[j][v])
#          phase_y.append(supercargo_inertia[j][v])
#     sphase_x.append(phase_x)
#     sphase_y.append(phase_y)    
#     phase_x=[]
#     phase_y=[]  
    
    """ Pairing the values for the phase space """
    
    for v in range(0,clusters):
#     print(v)   
     for j in range(0,iterations):
#          print(supercargo_centroid[j][v][0])
          phase_x.append(supercargo_centroid[j][v][0])
          phase_y.append(supercargo_centroid[j][v][1])
     sphase_x.append(phase_x)
     sphase_y.append(phase_y)    
     phase_x=[]
     phase_y=[]  
    
     """ Save output in JSON format via Pandas """
     phase_pd=pd.DataFrame({"phase_x":sphase_x[v],"phase_y":sphase_y[v]})
#     phase_pd.reset_index().to_json(orient='records',path_or_buf=target+dataset+'_Contour_'+str(s)+'_'+str(v)+'.json')

    sphase_x=[]
    sphase_y=[]
    supercargo_shifts=[]
    supercargo_inertia=[]
    supercargo_centroid=[]

""" Calculation of the density of points  """

dim = 60

xedges = np.linspace(np.min(seacluster_x),np.max(seacluster_x),dim)
yedges = np.linspace(np.min(seacluster_y),np.max(seacluster_y),dim)

H, xedges, yedges = np.histogram2d(seacluster_x, seacluster_y, bins=(xedges, yedges))
Hprima = H * 0
Hnew = H * 0

for i in range(0,dim-1):
    for j in range (0,dim-1):
        Hprima[i][j]=(H[j][i])

for k in range(0,dim-1):
    Hnew[k]=Hprima[(dim-2)-k]
    
""" Swap columns and rows for the D3.js required CSV / JSON format  """

newH = Hnew.ravel()
strH=list(newH)
#np.savetxt(target+dataset+'New_Volcano_2.csv',newH,fmt='%d',delimiter=",")    


my_df = pd.DataFrame(strH).T
my_df.to_csv(target+dataset+'_Volcano.csv', index=False, header=False)
#my_df.reset_index().to_json(orient='records',path_or_buf=target+'New_Volcano_2.json')


#with open(target+'New_Volcano_2.csv', "wb") as f:
#    writer = csv.writer(f)
#    writer.writerows(strH)
#np.savetxt(target+'New_Volcano_2.csv',strH,fmt='%d',delimiter=",")    
