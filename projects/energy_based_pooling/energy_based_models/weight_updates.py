"""
This module constains methods to update connection weights, that
can be used to re-configure an energy based pooler. 
The methods are expected to be set 
using `type.MethodType` or something similar.

Example:
    > pooler                = EnergyBasedPooler()
    > pooler.update_weights = MethodType(weight_updates.numenta_extended, pooler)

"""
import numpy as np

# The raw string is used because I don't want to escape special characters,
# so one can copy and paste the docstring into an environment which 
# is able to display LaTex.
def numenta_extended(self, X, Y):
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

        # Hebbian-like update
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


# The raw string is used because I don't want to escape special characters,
# so one can copy and paste the docstring into an environment which 
# is able to display LaTex.
def numenta_extended_bounded_by_zero(self, X, Y):
    r"""
    Method that updates the network's connections. The
    Weights are updated and computed according to

     - Visible to hidden: $ \Delta W_{ij} = y_i \cdot  ( \varepsilon_{\small{+}} \ x_j - \varepsilon_{\small{-}} \ \bar x_j ) $
     - Bias:              $ b_{i}  = B_b \cdot \alpha_i $
     - Hidden to hidden:  $ H_{ij} = B_H \cdot ( \alpha_{ij} - \alpha_i \ \alpha_j ) $ 
     - Hidden to hidden:  Bound below by zero, i.e. ensure $H_{ij} \geq 0$

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

        # Hebbian-like update
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
            # 
            if H[i,j] < 0:
                H[i,j] = 0.

    for i in range(n):
        H[i,i] = 0.






