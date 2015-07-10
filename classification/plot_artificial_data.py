#!/usr/bin/env python
# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2013, Numenta, Inc.  Unless you have an agreement
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

import csv
import os
import matplotlib.pyplot as plt
from settings import DATA_DIR, SIGNAL_TYPES, WHITE_NOISE_AMPLITUDE_RANGES



def findValidCSVNames():
  validFileNames = []
  for noiseAmplitude in WHITE_NOISE_AMPLITUDE_RANGES:
    for signalType in SIGNAL_TYPES:
      filePath = "%s/%s_%s.csv" %(DATA_DIR, signalType, noiseAmplitude)
      if os.path.exists(filePath):
        validFileNames.append(filePath)
        
  return validFileNames


csvFiles = findValidCSVNames()

plt.figure()
for filePath in csvFiles:
    with open(filePath, 'rb') as f:
      reader = csv.reader(f)
      headers = reader.next()
      #skip the 2 first rows
      reader.next()
      reader.next()
      x = []
      data = []
      labels = []
      for i, values in enumerate(reader):
        record = dict(zip(headers, values))
        x.append(i)
        data.append(record['y'])
        labels.append(record['label'])
  
      plt.subplot(len(csvFiles), 1, csvFiles.index(filePath) + 1)
      plt.plot(x, data)
      plt.title(filePath.split("/")[-1]) 
      plt.tight_layout()

plt.show()
