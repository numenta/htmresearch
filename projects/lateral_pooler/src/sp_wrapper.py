from nupic.algorithms.spatial_pooler import SpatialPooler 
import numpy as np


class SpatialPoolerWrapper(SpatialPooler):
    """
    This is a wrapper for the spatial pooler. 
    It just collects more statistics, otherwise behaves the exact same.
    """
    def __init__(self, **args):
        super(SpatialPoolerWrapper, self).__init__(**args)

        n = self._numColumns
        self.avg_activity_units = np.zeros(n)
        self.avg_activity_pairs = np.zeros((n,n))


    def compute(self, inputVector, learn, activeArray):
        """
        This method resembles the primary public method of the SpatialPooler class. 
        """
        super(SpatialPoolerWrapper, self).compute(inputVector, learn, activeArray)

        beta = 1 - 1/self._dutyCyclePeriod
        n    = self._numColumns
        Y    = activeArray.reshape((n,1))

        self.update_statistics_online(Y, beta)


    def update_statistics_online(self, Y, beta=0.9):
        """
        Updates the exponential moving averages over pairwise and individual 
        cell activities. 
        """
        P_pairs = self.avg_activity_pairs 
        P_units = self.avg_activity_units

        Q = Y * Y.T   

        P_pairs[:,:] = beta*P_pairs + (1-beta)*Q
        P_units[:]   = P_pairs.diagonal()

    @property
    def sparsity(self):
        return self._localAreaDensity

    @property
    def code_weight(self):
        return int(self._localAreaDensity*self._numColumns)
