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

import encoder_check
import numpy as np

from nupic.encoders.scalar import ScalarEncoder

if __name__ == "__main__":

  print "Testing ScalarEncoder Quality"

  maxval = 100.0
  minval = -100.0
  Nsamples = 1000

  encoder = ScalarEncoder(name="scalar", n=14, w=3, minval=minval,
                          maxval=maxval, periodic=True, forced=True)

  distance_function = lambda x,y : abs(x-y)

  sample_generator = lambda : np.random.uniform(minval, maxval)
  input_pairs_source = encoder_check.InputTripleCreator(sample_generator)

  err = encoder_check.encoderCheck(encoder, distance_function,
  	                               input_pairs_source)

  print "Average error: ",
  print err
