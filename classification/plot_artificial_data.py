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

import csv
import os

import matplotlib.pyplot as plt

from settings import (DATA_DIR, 
                      OUTFILE_NAME, 
                      WHITE_NOISE_AMPLITUDES, 
                      SEQUENCE_LENGTH)


def findValidCSVNames():
  validFileNames = []
  for noiseAmplitude in WHITE_NOISE_AMPLITUDES:
    for signalType in OUTFILE_NAME:
      filePath = "%s/%s_%s.csv" %(DATA_DIR, signalType, noiseAmplitude)
      if os.path.exists(filePath):
        validFileNames.append(filePath)
        
  return validFileNames


csvFiles = findValidCSVNames()

plt.figure()

for filePath in csvFiles:

  timesteps = []
  data = []
  labels = []
  categoriesLabelled = []
  with open(filePath, 'rb') as f:
      reader = csv.reader(f)
      headers = reader.next()

      #skip the 2 first rows
      reader.next()
      reader.next()

      for i, values in enumerate(reader):
        record = dict(zip(headers, values))
        timesteps.append(i)
        data.append(record['y'])
        labels.append(record['label'])
  
      ax = plt.subplot(len(csvFiles), 1, csvFiles.index(filePath) + 1)
      plt.plot(timesteps, data, label='signal')

      for k in range(len(timesteps) / SEQUENCE_LENGTH):
        if k%3 == 0:
          categoryColor = 'g'
        elif k%3 == 1:
          categoryColor = 'y'
        elif k%3 == 2:
          categoryColor = 'r'
        
        start = k*SEQUENCE_LENGTH
        end = (k+1) * SEQUENCE_LENGTH
       
        if categoryColor not in categoriesLabelled:
          label = 'sequence %s' %(k%3)
          categoriesLabelled.append(categoryColor)
        else:
          label=None
        plt.axvspan(start, end , facecolor=categoryColor, alpha=0.5, label=label)
      
      plt.xlim(xmin=0, xmax=len(timesteps))
      
      # title
      titleWords = filePath.split("/")[-1].replace('.csv', '').split("_")
      title = "%s amplitude = %s" %(' '.join(titleWords[:-1]), titleWords[-1])
      plt.title(title) 
      plt.tight_layout()
      
      plt.legend()
    
plt.show()
