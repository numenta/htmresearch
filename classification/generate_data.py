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

"""A simple script to generate a CSV with artificial data."""

import csv
import math
import random
from settings import CLASS_RANGES, DATA_DIR, SIGNAL_TYPES, WHITE_NOISE_AMPLITUDE, SIGNAL_AMPLITUDE, SIGNAL_MEAN

class _InvalidSignalName(Exception):
  pass
  
def generateData(whiteNoise=False, signal_mean=10, signal_period=100.0, number_of_points=4000, signal_amplitude=1, noise_amplitude=0.5, data_duplicates=2):

  if whiteNoise:
    fileName = "white_noise"
  else:
    fileName = "no_noise"
  
  if fileName not in SIGNAL_TYPES:
    raise _InvalidSignalName("File name should one of the following signal types %s but is '%s'" % (SIGNAL_TYPES, fileName))
  
  fileHandle = open("%s/%s.csv" % (DATA_DIR, fileName),"w")
  writer = csv.writer(fileHandle)
  writer.writerow(["x","y", "label"])

  for j in range(data_duplicates):
    for i in range(number_of_points):
      
      if whiteNoise:
        noise = noise_amplitude * random.random()
      else:
        noise = 0
  
      for label in CLASS_RANGES:
        for class_range in CLASS_RANGES[label]:
          start = class_range['start']
          end = class_range['end']
          if i>=start and i<=end:
            signal_modifier = 2 * (int(label[-1]) + 1)
            x = signal_modifier * (i * math.pi) / signal_period
            m1 = signal_modifier * signal_mean + signal_amplitude * math.sin(x) + noise    
  
            #print "Values labelled with class '%s' at %s" %(label, i)
            writer.writerow([x,m1, label])
    
  fileHandle.close()
  

if __name__ == "__main__":
  generateData(whiteNoise=False, signal_mean=SIGNAL_MEAN, signal_amplitude=SIGNAL_AMPLITUDE)
  generateData(whiteNoise=True, signal_mean=SIGNAL_MEAN, signal_amplitude=SIGNAL_AMPLITUDE, noise_amplitude=WHITE_NOISE_AMPLITUDE)
