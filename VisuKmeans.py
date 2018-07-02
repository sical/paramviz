#!/usr/bin/python

import sys

#from time import time
import matplotlib.pyplot as plt
import numpy as np
import random as rand

from sklearn import metrics
from sklearn.cluster import KMeans

import collections
import colorsys
import ntpath
import pandas as pd
from cycler import cycler


# TODO
# - Is that possible to modify the metric use in kmeans ?
# - Init by kmeans++ : http://ilpubs.stanford.edu:8090/778/1/2006-13.pdf

# Extensions
# http://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_stability_low_dim_dense.html#sphx-glr-auto-examples-cluster-plot-kmeans-stability-low-dim-dense-py

def simulte2DGaussianData(K, mean, cov, weight):
	'''
	Generate N data from a mixture of K gaussians with mean and cov parameters
	The weight array is the proportion of the Gaussains w.r.t the others
	'''

	N = 1000

	# Generate the class number for all N samples with proba given by weight
	classN = np.random.choice(K, N, p=weight)

	data2D = np.empty([N, 2], dtype=float)
	for n in range(N):
		# selected gaussian
		cl = classN[n]
		# sampling from the selected gaussian
		data2D[n]= np.random.multivariate_normal(mean[cl], cov[cl])
	
	return data2D

def GetData(filename) :
	'''
	Load a file of data with name filename
	Generate data according to a mixture of Gaussians if file does not exit.
	'''

	try:
		df=pd.read_csv(filename, sep='\t', header=None)
		print('Reading data')
		data2D = df.values
		pass
	except IOError:
		print ('Generating data')
		data2D = simulte2DGaussianData(K, mean, cov, weight)
		np.savetxt("./data/mixture2D.csv", data2D, delimiter="\t")

	return data2D

def allKmeans(data2D, NbClusters, nx, ny, ax, filename, RGB_tuples):
	'''
	Get cluster centers from initlisation of center according to a discrete grid, given by nx and ny.
	Plot centers according to the color in RGB-tuples
	'''

	marquershape = 'o'
	marquersize = 10 

	# Preparation for initial center loop position
	minX, maxX = min(data2D[:, 0]), max(data2D[:, 0])
	minY, maxY = min(data2D[:, 1]), max(data2D[:, 1])
	x = np.linspace(minX, maxX, nx)
	y = np.linspace(minY, maxY, ny)

	# Loop on init center position
	Centr_init= np.ndarray(shape=(NbClusters, 2), dtype=float)
	setInertia2 = set()
	listCenters2 = []
	count = np.zeros(10, dtype=int) # compte le nombre de kmeans pour chaque convergence
	for i1 in range(len(x)):
		for j1 in range(len(y)): 
			# first center coordinate
			Centr_init[0,:] = [x[i1], y[j1]]
			dot,  = ax.plot(x[i1], y[j1], color='green', marker=marquershape, markersize = marquersize+5)
			for i2 in range(len(x)):
				for j2 in range(len(y)): 

					Centr_init[1,:] = [x[i2], y[j2]]

					kmeans = KMeans(n_clusters=NbClusters, n_init =1, init=Centr_init, tol =0).fit(data2D)

					#on stoke la distance et la position des centres après convergence
					if kmeans.inertia_ not in setInertia2:
						listCenters2.append(kmeans.cluster_centers_)
					setInertia2.add(kmeans.inertia_)
					count[list(setInertia2).index(kmeans.inertia_)]+= 1
					#print(list(setInertia2).index(kmeans.inertia_))

					#print(list(setInertia2).index(kmeans.inertia_))
					ax.plot(x[i2], y[j2], color=RGB_tuples[list(setInertia2).index(kmeans.inertia_)], marker=marquershape, markersize = marquersize)
                    
            #name = "./figures/%s_%dx%d" % (ntpath.basename(filename), i1, j1) + ".png"
            #plt.savefig(name)
            #ax.lines.remove(dot)

	return list(setInertia2), listCenters2, count


if __name__ == "__main__":
	
	# Number of cluster. Only 2, but ust extended to whatever K
	NbClusters = 2

	# For generating data - Vector and matrix dimension should suit
	############################
	K = 3
	mean   = [[-1, -2],[2, 1],[0, 2]]
	cov    = [[[1, 0.1],[0.1, 2]], [[1.3, 0.6],[0.6, 2]], [[0.5, 0.3],[0.3, 2.5]]]
	weight = [0.2, 0.5, 0.3]
	# K = 2
	# mean   = [[-1, -1],[0, 2]]
	# cov    = [[[2, 0.1],[-0.3, 1]], [[1, 0.3],[0.3, 0.5]]]
	# weight = [0.4, 0.6]
	
	# Reading data (or generating if sys.argv[1] not present or erroneous)
	if len(sys.argv)>1:
		filename = sys.argv[1]
	else:
		filename = './data/mixture2D.csv'
	data2D = GetData(filename)
	# Number of 2D data
	N = data2D.shape[0]
	

	fig = plt.figure()
	ax = fig.add_subplot(1,1,1)
	# PLot of data in light grey
	ax.plot(data2D[:,0], data2D[:, 1], ls='none', marker='o', markersize=2, color=(0.75, 0.75, 0.75))
	# Plot of the center of convergence
	#for n in range(K):
	#	plt.plot(mean[n][0], mean[n][1], ls='none', marker='o', color='red')
	# shade of color for rendering
	#HSV_tuples = [(x*1.0/5, 0.5, 0.5) for x in range(5)]
	#RGB_tuples = list(map(lambda x: colorsys.hsv_to_rgb(*x), HSV_tuples))
	#print( RGB_tuples )
	RGB_tuples= list([(178/255, 18/255, 18/255), (255/255, 252/255, 25/255), (20/255, 133/255, 204/255), (0/255, 255/255, 70/255) ])
	#print( RGB_tuples )
	#input("Press Enter to continue...")

	# calcul de tous les kmeans sur la grille. On récupère kes valeurrs de distance différentes et les positions des centres
	nx, ny = 5,5
	listInertia2, listCenters2, count = allKmeans(data2D, NbClusters, nx, ny, ax, filename, RGB_tuples)
	print(count[0:len(listInertia2)])
	print(listInertia2)
	#print(listCenters2)
	#input("Press Enter to continue...")

	# #Possible classifications
	# print("Possible classifications")
	# #Classif= np.ndarray(shape=(N, len(listInertia2)), dtype=int)
	# for counter, centers in enumerate(listCenters2):
	# 	#print(counter, listCenters2[counter][: , 0])
	# 	fig2 = plt.figure()
	# 	ax2  = fig2.add_subplot(1,1,1)
	# 	Classif = KMeans(n_clusters=NbClusters, n_init =1, init=centers, tol =0).fit_predict(data2D)
	# 	print(Classif.dtype)
	# 	print(Classif, Classif[0], RGB_tuples[Classif[0]])
	# 	print(RGB_tuples[Classif])
	# 	p1 = ax2.plot(data2D[:,0], data2D[:, 1], color=RGB_tuples[Classif] )
	# 	p2 = ax2.plot(listCenters2[counter][0, 0], listCenters2[counter][0 , 1], color='red', marker='d', markersize = 20)
	# 	p3 = ax2.plot(listCenters2[counter][1, 0], listCenters2[counter][1 , 1], color='green', marker='d', markersize = 20)
	# 	name = "./figures/%s_classif%d" % (ntpath.basename(filename), counter) + ".png"
	# 	titre =  'Inertia '+ str(listInertia2[counter]) + ' - ' + str(count[counter])
	# 	plt.title( titre)
	# 	plt.savefig(name)
	# 	#input("Press Enter to continue...")
	# 	#plt.show()
	# 	fig2.clf()

	#print("Confusion matrices")
	#for counter in range(0, len(listInertia2)):
	#	print(metrics.confusion_matrix(Classif[counter%len(listInertia2), :], Classif[(counter+1)%len(listInertia2), :]))
	
	print("Result with init by kmeans++")
	fig2 = plt.figure()
	ax2  = fig2.add_subplot(1,1,1)
	#ax2.set_prop_cycle(cycler('color', ['c', 'm', 'y', 'k']) + cycler('lw', [1, 2, 3, 4]))
	kmeanspp = KMeans(n_clusters=NbClusters, n_init =1, init='k-means++', tol =0)
	classif    = kmeanspp.fit_predict(data2D)
	parameters = kmeanspp.fit(data2D)
	print(parameters.inertia_)
	print(parameters.cluster_centers_)
	print(classif.dtype)
	ax2.scatter(data2D[:,0], data2D[:, 1],  c=classif)
	titre = 'Init. par kmeans++ - Inertia ' + str(parameters.inertia_)
	plt.title(titre )
	name = "./figures/%s_classif_initkmeans++" % (ntpath.basename(filename)) + ".png"
	plt.savefig(name)



