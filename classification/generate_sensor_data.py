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

"""A simple script to generate a CSV with artificial data."""

import csv
import math
import os
import random
from settings import \
  SEQUENCE_LENGTH, \
  NUM_CATEGORIES, \
  DATA_DIR, \
  DEFAULT_WHITE_NOISE_AMPLITUDE, \
  WHITE_NOISE_AMPLITUDE_RANGES, \
  SIGNAL_AMPLITUDE, \
  SIGNAL_MEAN, \
  NUM_RECORDS, \
  SIGNAL_PERIOD


  
def generateData(dataDir=None, 
                 signal_mean=SIGNAL_MEAN, 
                 signal_period=SIGNAL_PERIOD, 
                 number_of_points=NUM_RECORDS, 
                 signal_amplitude=SIGNAL_AMPLITUDE, 
                 noise_amplitude=DEFAULT_WHITE_NOISE_AMPLITUDE):
  
  
  fileName = "white_noise_%s" %noise_amplitude
    
  if not dataDir:
    dataDir = DATA_DIR
  
  # make sure the directory exist. if not, create it.
  if not os.path.exists(dataDir):
    os.makedirs(dataDir)
  
  fileHandle = open("%s/%s.csv" % (dataDir, fileName),"wb")
  writer = csv.writer(fileHandle)
  writer.writerow(["x","y", "label"])
  writer.writerow(["float","float","int"])
  writer.writerow(["","","C"]) # C is for category. 
  # WARNING: if the C flag is forgotten in the dataset, then all records will be arbitrarily put
  # in the same category (i.e. category 0). So make sure to have the C flag -- otherwise
  # you'll get 100% classification accuracy regardless of the input data :-P


  endOfSequence = SEQUENCE_LENGTH
  label = 2
  for i in range(number_of_points):
    
    noise = noise_amplitude * random.random()
    
    if i == endOfSequence:
      endOfSequence += SEQUENCE_LENGTH
      if label == 0:
        label = 2
      else:
        label -= 1
      
    signal_modifier = 2 * (label + 1)
    x = signal_modifier * (i * math.pi) / signal_period
    m1 = signal_modifier * signal_mean + signal_amplitude * math.sin(x) + noise    

    writer.writerow([x,m1, label])
    #writer.writerow([i,i, label])
      
    
  fileHandle.close()
  
  print "Data generated. File saved to %s/%s.csv" % (dataDir, fileName)
  

if __name__ == "__main__":
  for whiteNoiseAmplitude in WHITE_NOISE_AMPLITUDE_RANGES:
    generateData(noise_amplitude=whiteNoiseAmplitude)
