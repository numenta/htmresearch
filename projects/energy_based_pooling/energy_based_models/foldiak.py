import numpy as np
from scipy.special import expit 
from scipy.optimize import fixed_point
from sparse_coding.utils import trim_doc

class SparseCoder(object):
    """
    Implementation of the network described in

        P. Földiák, Forming sparse representations by local anti-Hebbian learning,
        Biological Cybernetics 64 (1990), 165--170.

    Notation has been adapted from the paper.
    """

    def __init__(self, 
        inputSize  = 64, 
        outputSize = 16, 
        seed       = -1,
        alpha      =  0.1,
        beta       =  0.02,
        gamma      =  0.02,
        lam        = 10,
        p          = 0.125
        ):
        """
        Args
        ----
        inputSize: 
            Number of visible units
        outputSize:
            Number of hidden units
        seed:
            Random seed
        alpha:
            Learning rate for anti-Hebbian updates on W
        beta:
            Learning rate for Hebbian updates on Q
        lambda:
            Learning rate for threshold updates
        p:
            Desired activation probability of a unit

        """
        self._seed(seed)
        self._inputSize  = inputSize
        self._outputSize = outputSize
        self._alpha      = alpha
        self._beta       = beta
        self._gamma      = gamma
        self._lambda     = lam
        self._p          = p

        #-----------------------------------------
        # Q - Visible-to-hidden connections
        # W - Hidden-to-hidden connections 
        # t - Activity thresholds (bias)
        #-----------------------------------------

        self._Q          = np.zeros((self._outputSize, self._inputSize))
        self._W          = np.zeros((self._outputSize, self._outputSize))
        self._t          = np.ones(self._outputSize)*p
        self.initialize_weights()


    def _seed(self, seed=-1):
        """
        Initialize the random seed
        """
        if seed != -1:
          self._random = np.random.RandomState(seed)
        else:
          self._random = np.random.RandomState()


    def initialize_weights(self):
        """Randomly initializes the visible-to-hidden connections."""
        n = self._outputSize
        m = self._inputSize
        self._Q = self._random.sample((n,m))

        # Normalize the weights of each units
        for i in range(n):
            self._Q[i] /=  np.sqrt( np.dot(self._Q[i], self._Q[i]) )

     
    def encode(self, x):
        """
        Given an input array `x` it returns its associated encoding `y(x)`.
        Please cf. the paper for more details.

        Note that NO learning takes place.
        """
        n = self._outputSize
        y = np.zeros(n)
        Q = self._Q
        W = self._W
        t = self._t
        lam = self._lambda


        try:
            
            y_star = np.random.sample(n)
            y_star = fixed_point(lambda p: expit(lam * ( np.dot(Q,x) + np.dot(W,p) - t)), 
                                 y_star, maxiter=2000, method='del2')

        except RuntimeError:
            pass


        winner = np.where(y_star > 0.5)[0]
        y[ winner ] = 1.

        return y


    def encode_batch(self, X):
        """Encodes a whole batch of inputs."""
        X      = inputBatch
        encode = self.encode
        Y      = np.array([ encode(x) for x in X])
        return Y


    def learn(self, x):
        """
        Encodes an input and updates the weights of the network accordingly.
        Please cf. the paper for more details.
        """
        y = self.encode(x)
        self.update_weights(x,y)
        return y


    def update_weights(self, x, y):
        """Weight updates as described in the Földiák's paper."""
        m     = self._inputSize
        n     = self._outputSize
        W     = self._W
        Q     = self._Q
        t     = self._t
        alpha = self._alpha
        beta  = self._beta
        gamma = self._gamma
        p     = self._p

        for i in range(n):
            for j in range(n):
                Delta_W  = - alpha * ( y[i] * y[j] - p*p )
                W[i,j]  += Delta_W

                if i==j or W[i,j] > 0:
                    W[i,j] = 0.


        for i in range(n):
            for j in range(m):
                Delta_Q  = beta * y[i] * ( x[j] - Q[i,j] )
                Q[i,j]  += Delta_Q



        for i in range(n):
            Delta_t  = gamma * ( y[i] - p )
            t[i]    += Delta_t


    def __str__(self):
        """Returns a summary of the encoder."""
        
        summary = """

        Foldiak's Hopfield Net
        ----------------------
        
        Notation mostly adapted from the paper:

         - input size  : {self._inputSize}
         - output size : {self._outputSize}
         - alpha       : {self._alpha}
         - beta        : {self._beta}
         - gamma       : {self._gamma}
         - lambda      : {self._lambda}
         - p           : {self._p}

        ----------------------

        """
        summary = trim_doc(summary)
        summary = summary.format(self=self)
        return summary






