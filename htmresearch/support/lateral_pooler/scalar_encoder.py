# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2017, Numenta, Inc.  Unless you have an agreement
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
import math


class ScalarEncoder(object):
  """
  Quick implementation of a scalar encoder, that is, a map that
  encodes a real valued vector x as a binary vector as outlined below.

  Idea
  ----               
                                 x                 
                    |------------|------|
  
                   min                max     
                    
                      
      f((x,y)))  =  [. . . . . 1 1 1 . .] 

  """
  def __init__(self, min_value = 0., max_value=1., num_bits=1000, weight=20):
    self.min         = min_value
    self.max         = max_value
    self.num_bits    = num_bits
    self.weight      = weight
    self.bin_size    = 1/float(num_bits - weight)

  def __call__(self, x):
    return self.encode(x)

  def encode(self, x):
    n = self.num_bits
    w = self.weight 
    y = np.zeros(n)

    if x <= self.min:
      start = 0

    elif x >= self.max:
      start = n - w

    else:
      start = int( math.floor( (x - self.min) /self.bin_size))     

    end = start + w
    y[start:end] = 1.

    return y

