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
This module constains energy functions, that can be used 
to re-configure an energy based pooler. 
The methods are expected to be set 
using `type.MethodType` or something similar.

Example:
    > pooler        = EnergyBasedPooler()
    > pooler.energy = MethodType(energy_functions.numenta_extended, pooler)

"""

import numpy as np
from  numpy import dot, exp, maximum

# The raw string is used because I don't want to escape special characters,
# so one can copy and paste the docstring into an environment which 
# is able to display LaTex.
def numenta(self, x, y):
    r"""
    Numenta's energy:
    $$ 
        E(x,y)  = - \sum_i y_i \cdot \exp( - b_i ) \cdot (\sum_j W_{ij} \ x_j ) + S(y)
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
    energy       = - dot( y , exp( - b )  *  dot(W, x) ) 

    return energy + size_penalty 


# The raw string is used because I don't want to escape special characters,
# so one can copy and paste the docstring into an environment which 
# is able to display LaTex.
def numenta_extended(self, x, y):
    r"""
    Numenta's energy extended with an additional term (the H-term) 
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
def numenta_extended_no_size_penalty(self, x, y):
    r"""
    Numenta's energy with an additional term (the H-term) 
    to decorrelate pairwise column activity, but with NO size penalty:
    $$ 
        E(x,y)  = - \sum_i y_i \cdot \exp( - b_i - \sum_j H_{ij} \ y_j ) \cdot (\sum_j W_{ij} \ x_j ) 
    $$
    """
    pooler = self
    W      = pooler.connections.visible_to_hidden
    H      = pooler.connections.hidden_to_hidden
    b      = pooler.connections.hidden_bias
    n, m   = pooler.output_size, pooler.input_size
    w      = pooler.code_weight

    energy = - dot( y , exp( - b - dot(H, y) )  *  dot(W, x) ) 

    return energy 


 