#!/usr/bin/python

import numpy as np

# From http://flothesof.github.io/k-means-numpy.html

import matplotlib.pyplot as plt
import numpy as np

class KmeansTrace_standalone:
    def __init__(self, Data2D, K, c, grid):

        self.__Data2D = Data2D
        self.__K = K

        nx, ny = grid.get_nxny()
        x, y   = grid.get_xy()

        self.__Centres_init = np.ndarray(shape=(K, 2), dtype=float)
        for k in range(K):
            nb, reste = divmod(c[k], ny)
            self.__Centres_init[k,:] = [x[nb], y[reste]]

#            def initialize_centroids(self.__Data2D, k):
#                """returns k centroids from the initial points"""
#                centroids = points.copy()
#                np.random.shuffle(centroids)
#                return centroids[:k]

            def closest_centroid(self.__Data2D, self.__Centres_init):
                """returns an array containing the index to the nearest centroid for each point"""
                distances = np.sqrt(((points - centroids[:, np.newaxis])**2).sum(axis=2))
                return np.argmin(distances, axis=0)

            def move_centroids(self.__Data2D, closest, centroids):
                """returns the new centroids assigned from the points closest to them"""
                return np.array([points[closest==k].mean(axis=0) for k in range(centroids.shape[0])])     

	def getAll(self):
		return self.__labels_, self.__cluster_centers_, self.__inertia_, self.__best_n_iter_
	def getLabels(self):
		return self.__labels_
	def getCenters(self):
		return self.__cluster_centers_
	def getInertia(self):
		return self.self.__inertia_
	def getIter(self):
		return self.__best_n_iter_

	def getCenter_Init(self):
		return self.__Centres_init


#from sklearn.cluster import KMeans
#from sklearn.cluster import k_means
#
#class KmeansTrace:
#	def __init__(self, Data2D, K, c, grid):
#		
#		self.__Data2D = Data2D
#		self.__K = K
#
#		nx, ny = grid.get_nxny()
#		x, y   = grid.get_xy()
#		
#		self.__Centres_init = np.ndarray(shape=(K, 2), dtype=float)
#		for k in range(K):
#			nb, reste = divmod(c[k], ny)
#			self.__Centres_init[k,:] = [x[nb], y[reste]]
#
#		self.__cluster_centers_, self.__labels_, self.__inertia_, self.__best_n_iter_ = k_means(self.__Data2D, n_clusters=K, max_iter = 1000, n_init = 1, init=self.__Centres_init, tol= 1e-10, return_n_iter=True)
#
#
#	def getAll(self):
#		return self.__labels_, self.__cluster_centers_, self.__inertia_, self.__best_n_iter_
#	def getLabels(self):
#		return self.__labels_
#	def getCenters(self):
#		return self.__cluster_centers_
#	def getInertia(self):
#		return self.self.__inertia_
#	def getIter(self):
#		return self.__best_n_iter_
#		
#	def getCenter_Init(self):
#		return self.__Centres_init
#	

#    SAFE BACKUP

#    def initialize_centroids(points, k):
#        """returns k centroids from the initial points"""
#        centroids = points.copy()
#        np.random.shuffle(centroids)
#        return centroids[:k]
#
#    def closest_centroid(points, centroids):
#        """returns an array containing the index to the nearest centroid for each point"""
#        distances = np.sqrt(((points - centroids[:, np.newaxis])**2).sum(axis=2))
#        return np.argmin(distances, axis=0)
#
#    def move_centroids(points, closest, centroids):
#        """returns the new centroids assigned from the points closest to them"""
#        return np.array([points[closest==k].mean(axis=0) for k in range(centroids.shape[0])])
