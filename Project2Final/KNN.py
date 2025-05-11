__authors__ = ['1633672', '1634802', '1632753']
__group__ = 'DL.10, DJ.17'

import numpy as np
import math
import operator
from scipy.spatial.distance import cdist


class KNN:
    def __init__(self, train_data, labels):

        self._init_train(train_data)
        self.labels = np.array(labels)

    def _init_train(self, train_data):
        if not isinstance(train_data, float):
            self.train_data = train_data.astype(np.float)
        else:
            self.train_data = train_data

        tamaño = len(train_data)
        division = np.int64(np.divide(np.size(self.train_data), tamaño))
        self.train_data = np.reshape(self.train_data, (tamaño, division))

    def get_k_neighbours(self, test_data, k):
        principio = False
        if not isinstance(test_data, float):
            test_data = test_data.astype(np.float)

        tamaño = len(test_data)
        division = np.int64(np.divide(np.size(test_data), tamaño))
        test_data = np.reshape(test_data, (tamaño, division))

        self.neighbors = self.labels[np.argsort(cdist(test_data, self.train_data))[:, :k]]

        if not principio:
            principio = True

    def get_class(self):
        """
        Get the class by maximum voting
        :return: 2 numpy array of Nx1 elements.
                1st array For each of the rows in self.neighbors gets the most voted value
                            (i.e. the class at which that row belongs)
                2nd array For each of the rows in self.neighbors gets the % of votes for the winning class
        """

        output = {}
        lista = []

        for prenda in self.neighbors:

            prenda_output = {}
            for i in range(len(prenda)):
                if prenda[i] not in prenda_output:
                    prenda_output[prenda[i]] = 1
                else:
                    prenda_output[prenda[i]] += 1

            lista.append(max(prenda_output, key=prenda_output.get))

            for prenda, cantidad in prenda_output.items():
                if prenda not in output:
                    output[prenda] = cantidad
                else:
                    output[prenda] += cantidad
        prenda_percent = [output[prenda] / sum(output.values()) * 100 for prenda in output]

        return np.array(lista)

    def predict(self, test_data, k):
        """
        predicts the class at which each element in test_data belongs to
        :param test_data: array that has to be shaped to a NxD matrix ( N points in a D dimensional space)
        :param k:         :param k:  the number of neighbors to look at
        :return: the output form get_class (2 Nx1 vector, 1st the classm 2nd the  % of votes it got
        """
        self.get_k_neighbours(test_data, k)

        a = self.get_class()

        return a