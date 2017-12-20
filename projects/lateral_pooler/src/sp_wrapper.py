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

        beta = 1.0 - 1.0/self._dutyCyclePeriod
        n    = self._numColumns
        Y    = activeArray.reshape((n,1))

        self.update_statistics_online(Y, beta)

    def encode(self, X):
        d = X.shape[1]
        n = self._numColumns
        Y = np.zeros((n,d))
        for t in range(d):
            self.compute(X[:,t], False, Y[:,t])
            
        return Y

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
        inhibitionArea = ((2*self._inhibitionRadius + 1)
                                    ** self._columnDimensions.size)
        inhibitionArea = min(self._numColumns, inhibitionArea)
        density        = float(self._numActiveColumnsPerInhArea) / inhibitionArea
        return density

    @property
    def code_weight(self):
        return self._numActiveColumnsPerInhArea

    @property
    def feedforward(self):
        m = self._numInputs
        n = self._numColumns
        W = np.zeros((n, m))
        for i in range(self._numColumns):
            self.getPermanence(i, W[i, :])

        return W

    @property
    def shape(self):
        return self._numColumns, self._numInputs