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

import numpy as np

from nupic.data import SENTINEL_VALUE_FOR_MISSING_DATA
from nupic.encoders.base import Encoder
from nupic.encoders.scalar import ScalarEncoder



class SoundEncoder(Encoder):
  """
  This is an implementation of a sound encoder. A sound wave is converted into
  the maximum frequency detected according to FFT, and this frequency is
  encoded into an SDR using a ScalarEncoder.
  """


  def __init__(self, n, w, rate, chunk, minval=20, maxval=20000, name=None):
    """
    @param n int the length of the encoded SDR
    @param w int the number of 1s in the encoded SDR
    @param rate int the number of sound samples per second
    @param chunk int the number of samples in an input
    @param minval float the lowest possible frequency detected
    @param maxval float the highest possible frequency detected
    @param name string the name of the encoder
    """
    self.n = n
    self.w = w
    self.rate = rate
    self.chunk  = chunk
    self.minval = minval
    self.maxval = maxval
    self.name = name
    self._scalarEncoder = ScalarEncoder(name="scalar_"+str(name), n=n, w=w,
                                        minval=minval, maxval=maxval)


  def _detectFrequency(self, inputArr):
    """Use FFT to find maximum frequency present in the input."""
    fftData=abs(np.fft.rfft(inputArr))**2
    maxFreqIdx = np.argmax(fftData)

    if maxFreqIdx < len(fftData)-1:
      # Quadratic interpolation
      y0, y1, y2 = np.log(fftData[maxFreqIdx-1:maxFreqIdx+2:])
      x1 = (y2 - y0) * .5 / (2 * y1 - y2 - y0)
      return (maxFreqIdx+x1)*(self.rate/self.chunk)

    # Maximum idx is last in list, so cannot do quadratic interpolation
    return (maxFreqIdx+x1)*(self.rate/self.chunk)


  def encodeIntoArray(self, inputArr, output):
    if not isinstance(inputArr, (list, np.ndarray)):
      raise TypeError(
          "Expected a list or numpy array but got input of type %s" % type(inputArr))

    if inputArr == SENTINEL_VALUE_FOR_MISSING_DATA:
      output[0:self.n] = 0
    else:
      frequency = self._detectFrequency(inputArr)
      # Fail fast if frequency is outside allowed range.
      if (frequency < self.minval) or (frequency > self.maxval):
        raise ValueError(
             "Frequency value %f is outside allowed range (%f, %f)" % (
                  frequency, self.minval, self.maxval))

      output[0:self.n] = self._scalarEncoder.encode(frequency)


  def getWidth(self):
    return self.n
