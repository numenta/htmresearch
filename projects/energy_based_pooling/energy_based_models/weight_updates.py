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
def numenta(self, X, Y):
    r"""
    Method that updates the network's connections. 
    Numenta's classic update:

     - Visible to hidden: $ \Delta W_{ij} = y_i \cdot  ( \varepsilon_{\small{+}} \ x_j - \varepsilon_{\small{-}} \ \bar x_j ) $
     - Bias:              $ b_{i}  = B_b \cdot \alpha_i $

    """
    batchSize    = len(X)
    W            = self.connections.visible_to_hidden
    n, m         = W.shape
    bias         = self.connections.hidden_bias
    incr         = self.weight_incr
    decr         = self.weight_decr
    boost_bias   = self.boost_strength_bias
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

            if H[i,j] < 0:
                H[i,j] = 0.

    for i in range(n):
        H[i,i] = 0.


# The raw string is used because I don't want to escape special characters,
# so one can copy and paste the docstring into an environment which 
# is able to display LaTex.
def numenta_new_local_inhibition(self, X, Y):
    r"""
    Weight updates for the new local inhibition procedure (experimental and work in progress).

    See "Numenta’s local inhibition revisited" (Section 6) in `latex/notes.pdf`.

    Note that we directly encode the "activation probability" $a_{ij}$ as $h_{ij}$. 
    This is just a temporary hack. In Consequence, the hidden-to-hidden connections 
    are NOT symmetric anymore!
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
            H[i,j] =  alpha[i,j] /( np.sum( alpha[i,:] ) - alpha[i,i] )  

    for i in range(n):
        H[i,i] = 0.





