# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2016, Numenta, Inc.  Unless you have an agreement
# with Numenta, Inc., for a separate license for this software code, the
# following terms and conditions apply:
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero Public License version 3 as
# published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU Affero Public License for more details.
#
# You should have received a copy of the GNU Affero Public License
# along with this program.  If not, see http://www.gnu.org/licenses.
#
# http://numenta.org/licenses/
# ----------------------------------------------------------------------

import numpy as np
from   numpy import dot, exp, maximum
from   scipy.special import expit 
from   sparse_coding.utils import trim_doc


class Network(object):
    """Just a container class to collect the network connections."""
    def __init__(self, 
        visible_to_visible,
        visible_to_hidden,
        hidden_to_hidden,
        hidden_bias,
        visible_bias       
        ):
        self.visible_to_visible = visible_to_visible
        self.visible_to_hidden  = visible_to_hidden
        self.hidden_to_hidden   = hidden_to_hidden
        self.hidden_bias        = hidden_bias
        self.visible_bias       = visible_bias


class EnergyBasedPooler(object):
    """An energy-based spatial pooler implementation.

    Example Usage
    -------------
    ```
    # Instantiate
    coder = EnergyBasedPooler(...)

    # Re-configure 
    coder.energy         = MethodType(energy_functions.numenta_extended, coder)
    coder.update_weights = MethodType(weight_updates.numenta_extended_bounded_by_zero, coder)

    # Train
    for line in file:
        input_vector = numpy.array(line)
        coder.learn(input_vector)

    ```

    """

    def __init__(self, 
                 inputSize           = 100, 
                 outputSize          = 128, 
                 codeWeight          = 4, 
                 seed                = -1,
                 smoothingPeriod     = 50., 
                 boostStrengthBias   = 100.,
                 boostStrengthHidden = 100.,
                 learningRateHidden  = 0.01,
                 learningRateBias    = 0.01,
                 weightIncr          = 0.01,
                 weightDecr          = 0.01
        ):
        """
        Args
        ----
        input_size: 
            Number of visible units
        output_size:
            Number of hidden units
        code_weight:
            Number of desired active output units
        seed:
            Random seed
        smoothing_period:
            The inverse describes the smoothing parameter in 
            for the exponential moving average
        boost_strength_bias:
            The factor that controls the impact of the bias on the energy
        boost_strength_hidden:
            The factor that controls the impact of the hidden-to-hidden 
            connections on the energy
        learning_rate_bias:
            Learning rate for bias updates 
            (for the standard configuration we don't make use of this)
        learning_rate_hidden:
            Learning rate for the hidden-to-hidden updates
            (for the standard configuration we don't make use of this)
        weight_incr:
            Learning rate of the positive term of the Hebbian update 
            performed on the visible-to-hidden connections
        weight_decr:
            Learning rate of the negative term of the Hebbian update 
            performed on the visible-to-hidden connections

        """

        if seed != -1:
          self._random = np.random.RandomState(seed)
        else:
          self._random = np.random.RandomState()

        self.input_size  = inputSize
        self.output_size = outputSize
        self.code_weight = codeWeight
        self.sparsity    = float(codeWeight)/float(outputSize)
        self.shape       = (self.output_size, self.input_size)

        # ------------------------
        #  The connection weights 
        #  of the network
        # ------------------------
        self.connections = Network(
            visible_to_visible = None,
            visible_to_hidden  = np.zeros((self.output_size, self.input_size)),
            hidden_to_hidden   = np.ones((self.output_size, self.output_size)),
            hidden_bias        = np.zeros(self.output_size),
            visible_bias       = None
        )
        self.initialize_connections()

        # -----------------
        #  Boost strengths
        # -----------------
        self.boost_strength_bias   = boostStrengthBias
        self.boost_strength_hidden = boostStrengthHidden
        
        # ----------------
        #  Learning rates
        # ----------------
        self.smoothing_period     = smoothingPeriod
        self.learning_rate_bias   = learningRateBias
        self.learning_rate_hidden = learningRateHidden
        self.weight_incr          = weightIncr
        self.weight_decr          = weightDecr

        # ----------------------------------------------------
        # Initialize avg. activities with optimal values, i.e. 
        #  - maximizing mean individual entropy, 
        #  - maximizing mean pairwise entropy
        # ----------------------------------------------------
        self.average_activity = np.ones((outputSize, outputSize))*self.sparsity*self.sparsity;
        for i in range(self.output_size):
            self.average_activity[i,i] = self.sparsity


    def initialize_connections(self):
        """
        Randomly initializes the visible-to-hidden connections.
        
        Each weight is independently sampled from a uniform distribution $U(0,1)$.
        The weights are NOT normalized!
        """
        n, m = self.connections.visible_to_hidden.shape
        self.connections.visible_to_hidden = np.zeros((n, m))

        for i in range(n):
            random_vector = self._random.rand(m)
            # norm          = np.sqrt( np.sum(random_vector * random_vector) )
            self.connections.visible_to_hidden[i]  = random_vector


    def compute(self, inputVector, learn, activeArray):
        """This method resembles the primary public method of the SpatialPooler class. 
        It takes a input vector and outputs the indices of the active columns. If 'learn' 
        is set to True, this method also performs weight updates and updates to the activity 
        statistics according to the respective methods implemented below."""
        x = inputVector
        y = self.encode(x)
        active_units = np.where(y==1.)[0]
        
        if learn:
            self.update_statistics([y]) 
            self.update_weights([x],[y])

        activeArray[active_units] = 1.
        return active_units


    def encode(self, x):
        """
        Given an input array `x` it returns its associated encoding `y(x)`, that is, 
        a stable configuration (local energy minimum) of the hidden units 
        while the visible units are clampled to `x`.
        
        Note that NO learning takes place.
        """
        E      = self.energy
        y_min  = self.find_energy_minimum(E, x)
        return y_min


    def encode_batch(self, inputBatch):
        """Encodes a whole batch of input arrays, without learning."""
        X      = inputBatch
        encode = self.encode
        Y      = np.array([ encode(x) for x in X])
        return Y

    def learn(self, x):
        """Encodes an input array, and performs weight updates and updates to the activity 
        statistics according to the respective methods implemented below."""
        y = self.encode(x)
        self.update_statistics([y]) 
        self.update_weights([x],[y])
        return y

    def learn_batch(self, inputBatch):
        """Encodes a whole batch of input arrays, and performs weight updates and updates to the activity 
        statistics according to the respective methods implemented below."""
        X = inputBatch
        Y = self.encode_batch(X)
        self.update_statistics(Y) 
        self.update_weights(X,Y)
        return Y

    def update_statistics(self, activityVectors):
        """Updates the variable that maintains exponential moving averages of 
        individual and pairwise unit activiy"""
        Y = activityVectors
        n = self.output_size
        A = np.zeros((n, n))
        batchSize = len(Y)

        for y in Y:
            active_units = np.where( y == 1 )[0]
            for i in active_units:
                for j in active_units:
                    A[i,j] += 1.

        A = A/batchSize
        self.average_activity = self.exponential_moving_average(self.average_activity, A, self.smoothing_period)

    # The raw string is used because I don't want to escape special characters,
    # so one can copy and paste the docstring into an environment which 
    # is able to display LaTex.
    def find_energy_minimum(self, energy, x, maxSteps=40000):
        r"""
        Naive hill descend algorithm:
        Starts at $y=0$ and iteratively chooses the direction of steepest descent from 
        a neighbourhood of the current position. If not indicated otherwise the neighbourhood
        consists of elements at Hamming distance less than or equal to $1$. 
        """
        E = energy
        y = np.zeros(self.output_size) 

        min_so_far = E(x, y)

        for _ in range(maxSteps):
            neighbours       = self.get_neighbours(y)
            energies         = np.array([ E(x,z) for z in neighbours ])
            steepest_descent = np.argmin(energies)

            if energies[steepest_descent] < min_so_far:
                y          = neighbours[steepest_descent]
                min_so_far =   energies[steepest_descent]
            else:
                break

        return y

    # The raw string is used because I don't want to escape special characters,
    # so one can copy and paste the docstring into an environment which 
    # is able to display LaTex.
    @staticmethod
    def get_neighbours(y):
        r"""
        Returns a neigbourhood $N$ of $y$ consisting of elements that 
        share $|y| - 1$ bits with $y$, i.e 
        $$
                    N(y) = \{ x : x^T \cdot y  = |y| - 1  \}
        $$
        """
        onBits     = np.where( y == 1. )[0]
        offBits    = np.where( y == 0. )[0]
        neighbours = [y.copy()]

        for on in onBits:
            z     = y.copy()
            z[on] = 0.
            neighbours.append(z)

        for off in offBits:
            z      = y.copy()
            z[off] = 1.
            neighbours.append(z)

        # np.random.shuffle(neighbours)

        return neighbours


    # The raw string is used because I don't want to escape special characters,
    # so one can copy and paste the docstring into an environment which 
    # is able to display LaTex.
    def energy(self, x, y):
        r"""
        Numenta's energy with an additional term (the H-term) 
        to decorrelate pairwise column activity:
        $$ 
            E(x,y)  = - \sum_i y_i \cdot \exp( - b_i - \sum_j H_{ij} \ y_j ) \cdot (\sum_j W_{ij} \ x_j ) + S(y)
        $$
        where the size size penalty is given by
        $$ 
            S(y) = \begin{cases}
                    0        &  \text{if $\|y\| \leq w$, and} \\
                    +\infty  &  \text{otherwise,}
                   \end{cases}
        $$ 
        """
        pooler = self
        W      = pooler.connections.visible_to_hidden
        H      = pooler.connections.hidden_to_hidden
        b      = pooler.connections.hidden_bias
        n, m   = pooler.output_size, pooler.input_size
        w      = pooler.code_weight

        size_penalty = 0 if dot(y,y) <= w else np.inf
        energy       = - dot( y , exp( - b - dot(H, y) )  *  dot(W, x) ) 

        return energy + size_penalty


    # The raw string is used because I don't want to escape special characters,
    # so one can copy and paste the docstring into an environment which 
    # is able to display LaTex.
    def update_weights(self, X, Y):
        r"""
        Method that updates the network's connections. The
        Weights are updated and computed according to

         - Visible to hidden: $ \Delta W_{ij} = y_i \cdot  ( \varepsilon_{\small{+}} \ x_j - \varepsilon_{\small{-}} \ \bar x_j ) $
         - Bias:              $ b_{i}  = B_b \cdot \alpha_i $
         - Hidden to hidden : $ H_{ij} = B_H \cdot ( \alpha_{ij} - \alpha_i \ \alpha_j ) $

        """
        batchSize    = len(X)
        W            = self.connections.visible_to_hidden
        H            = self.connections.hidden_to_hidden
        n, m         = W.shape
        bias         = self.connections.hidden_bias
        incr         = self.weight_incr
        decr         = self.weight_decr
        boost_bias   = self.boost_strength_bias
        boost_hidden = self.boost_strength_hidden
        alpha        = self.average_activity

        #---------------------------
        # visible-to-hidden updates
        #---------------------------
        for i in range(batchSize):
            y     = Y[i]
            x     = X[i]
            x_bar = np.ones(m) - x   

            W[ np.where(y == 1)[0] ] += incr*x - decr*x_bar 

        # Clip the visible-to-hidden connections 
        # to be between $0$ and $1$ 
        tooSmall = np.where(W < 0.)
        tooBig   = np.where(W > 1.)
        W[ tooSmall ] = 0.
        W[ tooBig   ] = 1.


        #---------------
        # (Hidden) Bias
        #---------------
        for i in range(n):
            bias[i] = boost_bias * alpha[i,i]

        #--------------------------
        # Hidden-to-hidden updates
        #--------------------------
        for i in range(n):
            for j in range(n):
                H[i,j] =  boost_hidden * (alpha[i,j]  -  alpha[i,i]*alpha[j,j])

        for i in range(n):
            H[i,i] = 0.


    @staticmethod
    def exponential_moving_average(oldAverage, newInput, period):
        """
        Exponential moving average with 
        smoothing factor alpha = 1/period

                        (period - 1)*oldAverage + newValue
          newAverage := ----------------------------------
                                    period

        See https://en.wikipedia.org/wiki/Exponential_smoothing 
        """
        assert(period >= 1)
        return (oldAverage * (period -1.0) + newInput) / period


    def __str__(self):
        """
        Returns a brief summary of the encoder and how it is configured,
        including which energy and weight updates are used, and 
        how the energy minimum is found.
        """
        summary = """

        Energy-based Pooler
        ---------------
        
        Configuration...

         - input size  : {self.input_size}
         - output size : {self.output_size}
         - code weight : {self.code_weight}
         - sparsity    : {self.sparsity}
         - Boost_b     : {self.boost_strength_bias}
         - Boost_H     : {self.boost_strength_hidden}
         - incr, decr  : {self.weight_incr}, {self.weight_decr}
         - LR_hidden   : {self.learning_rate_hidden}
         - LR_bias     : {self.learning_rate_bias}
        
        Docstrings...
        
        **initialize connections**
        {init_weights}

        **encode**
        {encode_doc}

        **update weights:**
        {update_doc}
        
        **energy:**
        {energy_doc}

        **find energy minimum:**
        {min_doc}

        ---------------

        """
        summary = trim_doc(summary)
        summary = summary.format(self=self, 
                                 init_weights = trim_doc(self.initialize_connections.__doc__),
                                 encode_doc = trim_doc(self.encode.__doc__),
                                 energy_doc = trim_doc(self.energy.__doc__),
                                 update_doc = trim_doc(self.update_weights.__doc__),
                                 min_doc    = trim_doc(self.find_energy_minimum.__doc__))
        return summary



