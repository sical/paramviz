#!/usr/bin/python

import numpy as np
import itertools

class DataGrid:
	def __init__(self, Data2D, nx, ny, K):
		self.__nx = nx
		self.__ny = ny

		# Preparation for initial center loop position
		self.__minX, self.__maxX = min(Data2D[:, 0]), max(Data2D[:, 0])
		self.__minY, self.__maxY = min(Data2D[:, 1]), max(Data2D[:, 1])
		self.__x = np.linspace(self.__minX, self.__maxX, self.__nx)
		self.__y = np.linspace(self.__minY, self.__maxY, self.__ny)

		# Liste de toutes les cases de la grille (en evitant les points symétriquees et les centres identiques)
		self.__grille = list(itertools.combinations(range(self.__nx*self.__ny), K))


	def get_nxny(self):
		return self.__nx, self.__ny

	def get_xy(self):
		return self.__x, self.__y

	def get_minXmaxX(self):
		return self.__minX, self.__maxX

	def get_minYmaY(Ylf):
		return self.__minY, self.__maxY

	def get_grille(self):
		return self.__grille
