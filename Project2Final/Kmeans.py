__authors__ = ['1632753', '1633672', '1634802']
__group__ = ['DJ.17', 'DL.10']

import numpy as np
from sklearn.utils.extmath import stable_cumsum

import utils
from sklearn.metrics.pairwise import cosine_distances


class KMeans:

    def __init__(self, X, K=1, options=None):
        """
         Constructor of KMeans class
             Args:
                 K (int): Number of cluster
                 options (dict): dictºionary with options
            """
        self.num_iter = 0
        self.K = K
        self._init_X(X)
        self._init_options(options)

    def _init_X(self, X):
        """Initialization of all pixels, sets X as an array of data in vector form (PxD)
            Args:
                X (list or np.array): list(matrix) of all pixel values
                    if matrix has more than 2 dimensions, the dimensionality of the sample space is the length of
                    the last dimension
        """

        if isinstance(X, float) is False:
            self.X = X.astype(np.float)
        if len(X[0]) != 3:
            val = np.int64(np.divide(np.size(self.X), 3))
            self.X = np.reshape(self.X, (val, 3))

    def _init_options(self, options=None):
        """
        Initialization of options in case some fields are left undefined
        Args:
            options (dict): dictionary with options
        """
        if options is None:
            options = {}
        if 'km_init' not in options:
            options['km_init'] = 'first'
        if 'verbose' not in options:
            options['verbose'] = False
        if 'tolerance' not in options:
            options['tolerance'] = 0.00315
        if 'max_iter' not in options:
            options['max_iter'] = np.inf
        if 'fitting' not in options:
            options['fitting'] = 'WCD'

        self.options = options

    def _init_centroids(self):
        """
        Initialization of centroids
        """
        ###########################################################
        #          MODIFY THE CODE BELOW THIS POINT               #
        ###########################################################
        first = False
        rand = False
        custom = False
        self.old_centroids = np.empty([self.K, 3])
        self.centroids = np.empty([self.K, 3], dtype=np.float64)
        self.centroids[:] = np.nan
        if self.options['km_init'].lower() == 'custom':
            custom = True
            i = 0
            np.random.seed()
            aux = np.random.randint(low=0, high=len(self.X) - 1)
            self.centroids[0] = self.X[aux]
            closest_dist_sq = cosine_distances(self.centroids[0, np.newaxis], self.X)[0] ** 2
            current_pot = closest_dist_sq.sum()

        elif self.options['km_init'].lower() == 'first':
            first = True
            i = 0
        elif self.options['km_init'].lower() == 'random':
            rand = True
            i = 0
            np.random.seed()
            aux = np.random.randint(low=0, high=len(self.X) - 1)

        if custom:
            for c in range(1, len(self.centroids)):
                rand_vals = np.random.seed() * current_pot
                candidate_ids = np.searchsorted(stable_cumsum(closest_dist_sq), rand_vals)
                self.centroids[c] = self.X[candidate_ids].toarray()
                new_dist_sq = cosine_distances(self.X[candidate_ids, :], self.X)[0] ** 2
                closest_dist_sq = np.minimum(new_dist_sq, closest_dist_sq)
                current_pot = closest_dist_sq.sum()

        if first:
            for pixel in self.X:
                if not any((np.equal(pixel, self.centroids).all(1))):
                    self.centroids[i] = pixel
                    i = np.add(i, 1)
                    if i == self.K:
                        break
        elif rand:
            while i != self.K:
                if not any((np.equal(self.X[aux], self.centroids).all(1))):
                    self.centroids[i] = self.X[aux]
                    i = np.add(i, 1)
                np.random.seed()
                aux = np.random.randint(low=0, high=len(self.X) - 1)

    def get_labels(self):
        """Calculates the closest centroid of all points in X
        and assigns each point to the closest centroid
        """
        self.labels = np.argmin(distance(self.X, self.centroids), axis=1)

    def get_centroids(self):
        """
        Calculates coordinates of centroids based on the coordinates of all the points assigned to the centroid
        """
        self.old_centroids = np.array(self.centroids, copy=True)
        result = []
        for k in range(self.K):
            sum_of_points = self.X[self.labels == k].sum(0)
            result.append(sum_of_points)

        bincount = np.bincount(self.labels).reshape(-1, 1)
        centroids = np.divide(result, bincount)
        self.centroids = centroids

    def converges(self):
        """
        Checks if there is a difference between current and old centroids
        """
        return np.allclose(self.old_centroids, self.centroids, rtol=self.options['tolerance'])

    def fit(self):
        """
        Runs K-Means algorithm until it converges or until the number
        of iterations is smaller than the maximum number of iterations.
        """
        self._init_centroids()

        while not self.converges() and self.num_iter < self.options['max_iter']:
            self.get_labels()
            self.get_centroids()
            self.num_iter = np.add(self.num_iter, 1)

    def withinClassDistance(self):
        """
         returns the whithin class distance of the current clustering
        """

        return np.multiply(np.divide(1, len(self.X)),
                           np.sum(np.square(np.amin(distance(self.X, self.centroids), axis=1))))

    def find_bestK(self, max_K):
        self.K = 2
        self.fit()
        wcd = self.withinClassDistance()
        for kn in range(3, max_K):
            self.K = kn
            self.fit()
            wcd_old = wcd
            wcd = self.withinClassDistance()
            if 0.2 > np.subtract(1, np.divide(wcd, wcd_old)):
                self.K = np.subtract(self.K, 1)
                self.fit()
                break

    def calculate_interClassDistance(self):
        total_distance = 0
        for centroid_idx, centroid in enumerate(self.centroids):
            centroid_pixels = np.where(self.labels == centroid_idx)[0]
            other_centroids = self.centroids[np.where(np.arange(len(self.centroids)) != centroid_idx)]
            for other_centroid in other_centroids:
                centroid_distance = np.sum((self.X[centroid_pixels] - other_centroid) ** 2)
                total_distance += centroid_distance
        average_distance = total_distance / len(self.X)
        return average_distance

    def fisherDiscriminant(self):

        return self.withinClassDistance() / self.calculate_interClassDistance()

    def find_bestKBetter(self, max_K, hType):
        self.K = 2
        self.fit()

        if hType == 'Fisher':
            wcd_old = self.fisherDiscriminant()
        elif hType == 'Inter':
            wcd_old = self.calculate_interClassDistance()
        else:
            wcd_old = self.withinClassDistance()

        self.K += 1
        find = False
        threshold = 10

        while (self.K <= max_K) and (find is False):
            self.fit()

            if hType == 'Fisher':

                wcd = self.fisherDiscriminant()
                percent = (wcd / wcd_old) * 100
            elif hType == 'Inter':

                wcd = self.calculate_interClassDistance()
                percent = (wcd_old / wcd) * 100
            else:

                wcd = self.withinClassDistance()
                percent = (wcd / wcd_old) * 100

            if 100 - percent < threshold:

                self.K -= 1
                find = True
            else:

                self.K += 1
                wcd_old = wcd

        if find is False:

            self.K = max_K
        self.fit()


def distance(X, C):
    term1 = np.square(np.subtract(X[:, 0, np.newaxis], C[:, 0]))
    term2 = np.square(np.subtract(X[:, 1, np.newaxis], C[:, 1]))
    term3 = np.square(np.subtract(X[:, 2, np.newaxis], C[:, 2]))
    dist = np.sqrt(np.add(np.add(term1, term2), term3))
    return dist


def get_colors(centroids):
    color_probs = utils.get_color_prob(centroids)
    labels = np.empty(len(centroids), dtype=object)
    for i in range(len(centroids)):
        labels[i] = utils.colors[np.argmax(color_probs[i])]
    return labels

