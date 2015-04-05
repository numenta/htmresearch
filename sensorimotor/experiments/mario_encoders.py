#!/usr/bin/env python
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

from nupic.encoders.sdrcategory import SDRCategoryEncoder



class MarioEncoders(object):
  
  def __init__(self, n, w):
    self.n = n
    self.w = w
  
  
  def _transformBinaryVector(vector):
    transform = [i for i in range(len(vector)) if vector[i]==1]
    assert sum(vector)==len(transform)
    return transform


  def encodeSensorySequences(self, sequences, n, w):
    SDRs = []
    for s in sequence:
      sdr = self._transformBinaryVector(SDRCategoryEncoder.encode(s, n, w, force=True))
      SDRs.append(sdr)
    return SDRs


  def encodeXYPositionSequences(self, x_sequences, y_sequences):
    # Encode both the x and y sequences for n/2 and w/2, then merge.
    assert len(x_sequences)==len(y_sequences)
    n = self.n/2
    w = self.w/2
    SDRs = []
    for i in range(len(x_sequences)):
      assert len(x_sequences[i])==len(y_sequences[i])
      x_sdr = self.encodeSensorySequences(x_sequences[i], n, w)
      y_sdr = self.encodeSensorySequences(y_sequences[i], n, w)
      sdr = x_sdr + y_sdr
      SDRs.append(sdr)
    return SDRs


  def encodeMotorSequences(self, sequences):
    # Note the encoding may end up with <w ON bits due to rounding down
    SDRs = []
    for sequence in sequences:
      SDR = []
      for s in sequence:
        section = n/len(s)
        numON = w/sum(s)  # for every ON bit in s, this is the # of ON bits in the subsequent sdr
        idx = [i*section for i in xrange(len(s)) if s[i]==1]
  #      sdr = numpy.zeros(n)
  #      for i in idx:
  #        sdr[i:i+numON] = 1
  #      sdr = self._transformBinaryVector(sdr)
        SDR.append(idx)
      SDRs.append(SDR)
    return SDRs
