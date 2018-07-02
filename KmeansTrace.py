#!/usr/bin/python

import numpy as np

#from sklearn.cluster import KMeans
from sklearn.cluster import k_means


class KmeansTrace:
	def __init__(self, Data2D, K, c, grid):
		
		self.__Data2D = Data2D
		self.__K = K

		nx, ny = grid.get_nxny()
		x, y   = grid.get_xy()
		
		self.__Centres_init = np.ndarray(shape=(K, 2), dtype=float)
		for k in range(K):
			nb, reste = divmod(c[k], ny)
			self.__Centres_init[k,:] = [x[nb], y[reste]]

		self.__cluster_centers_, self.__labels_, self.__inertia_, self.__best_n_iter_ = k_means(self.__Data2D, n_clusters=K, max_iter = 1000, n_init = 1, init=self.__Centres_init, tol= 1e-10, return_n_iter=True)


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
	
	# def __repr__(self):
	# 	return str(self)
	# def __str__(self):
	# 	return("nb iter = " + str(self.__best_n_iter_))
