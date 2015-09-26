#!/usr/bin/env python
# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2015, Numenta, Inc.  Unless you have an agreement
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
from suite import DistributedEncoder

num_encodings = 10
num_multiple_prediction = 2
nDim = 10

enc = DistributedEncoder(nDim,
                         minValue=-0.5,
                         maxValue=0.5,
                         classifyWithRandom=False)

for i in range(num_encodings):
  enc.encode(i)

symbolList = enc.encodings.keys()

outcome = []
for _ in range(500):
  # select two symbols, and decode using the average of the two encodings
  symbol1 = np.random.choice(symbolList)
  symbol2 = np.random.choice(symbolList)
  encoding1 = enc.encode(symbol1)
  encoding2 = enc.encode(symbol2)
  closest_symbols = enc.classify((encoding1+encoding2)/2, num=2)
  correct = symbol1 in closest_symbols and symbol2 in closest_symbols
  outcome.append(correct)
  print _, symbol1, symbol2, closest_symbols, correct

print " Decode Success Rate: ", np.mean(outcome)


