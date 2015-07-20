# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2015, Numenta, Inc.  Unless you have an agreement
# with Numenta, Inc., for a separate license for this software code, the
# following terms and conditions apply:
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3 as
# published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see http://www.gnu.org/licenses.
#
# http://numenta.org/licenses/
# ----------------------------------------------------------------------

"""
This script demonstrates how encoder_check.encoderCheck is used.

The example shows that a RandomDistributedScalarEncoder with higher
resolution will more tightly preserve the distance metric of the scalar
input space.

For three scalar values x, y, z, and their encodings Sx, Sy, and Sz, if
the overlap of Sx and Sy is greater than the overlap of Sx and Sz, we would
hope that the distance between x and y is less than the distance between x and
z. This is the logic that the encoderCheck employs. If it finds values that
violate this property, it reports it with a warning.
"""

import encoder_check
import numpy as np

from nupic.encoders.random_distributed_scalar import (
  RandomDistributedScalarEncoder
)

if __name__ == "__main__":

  print "Testing RSDE Quality"

  maxval = 100.0
  minval = -100.0
  Nsamples = 1000

  encoder1 = RandomDistributedScalarEncoder(name="encoder", resolution=1.0,
                                            w=23, n=500, offset=0.0)

  encoder2 = RandomDistributedScalarEncoder(name="encoder", resolution=10.0,
                                            w=23, n=500, offset=0.0)


  distance_function = lambda x,y : abs(x-y)

  sample_generator = lambda : np.random.uniform(minval, maxval)
  input_pairs_source = encoder_check.InputTripleCreator(sample_generator)

  err1 = encoder_check.encoderCheck(encoder1, distance_function,
                                    input_pairs_source)

  err2 = encoder_check.encoderCheck(encoder2, distance_function,
                                    input_pairs_source)

  print
  print "Warning rate for encoder w/ resolution 1.0: ",
  print err1
  print "Warning rate for encoder w/ resolution 10.0: ",
  print err2
