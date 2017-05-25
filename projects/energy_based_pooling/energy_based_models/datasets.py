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
from energy_based_models.scalar_encoder import ScalarEncoder 

def n_to_one(num_inputs, num_streams, bits_per_axis, weight_per_axis):
    """
    Creating inputs of the form: 

        ``( x, y, ..., y )''
              |---------|
                   n           
    Here n = num_streams -1.
    To be more precise, for each component we allocate a fixed number
    of units `bits_per_axis` and encode each component with a scalar encoder.

    This is a toy example for the following scenario: We are given 4 input streams, 
    such that 3 mutually determine each other, and the remaining one is independent from the rest.

    """
    num_bits = num_streams*bits_per_axis
    # Setting up the scalar encoder
    encoder_params = {
        "dimensions"      : num_streams, 
        "max_values"      : [[0.,1.]]*num_streams,
        "bits_per_axis"   : [bits_per_axis]*num_streams, 
        "weight_per_axis" : [weight_per_axis]*num_streams,
        "wrap_around"     : False
    }
    scalar_encode = ScalarEncoder(**encoder_params)

    xs = np.random.sample(num_inputs)
    ys = np.random.sample(num_inputs)
    input_vectors = np.zeros((num_inputs, num_bits))

    for i in range(num_inputs):
        input_vectors[i] = scalar_encode([xs[i]] + [ys[i]]*(num_streams - 1) )
    
    return input_vectors


def three_to_one(num_inputs, bits_per_axis, weight_per_axis):
    """
    Creating inputs of the form: 

        ``( x, y, y, y )''

    To be more precise, for each component we allocate a fixed number
    of units `bits_per_axis` and encode each component with a scalar encoder.

    This is a toy example for the following scenario: We are given 4 input streams, 
    such that 3 mutually determine each other, and the remaining one is independent from the rest.

    """
    return n_to_one(num_inputs, 4, bits_per_axis, weight_per_axis)