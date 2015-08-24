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

"""A simple script to generate a CSV with artificial data."""

import csv
import math
import os
import random

from settings import (SEQUENCE_LENGTH,
                      DATA_DIR,
                      OUTFILE_NAME,
                      WHITE_NOISE_AMPLITUDES,
                      SIGNAL_AMPLITUDE,
                      SIGNAL_MEAN,
                      NUM_RECORDS,
                      SIGNAL_PERIOD)



def generateData(dataDir,
                 outputFileName,
                 signalMean,
                 signalPeriod,
                 sequenceLength,
                 numPoints,
                 signalAmplitude,
                 numCategories,
                 noiseAmplitude):
  """
  Generate the artificial sensor data.
  @param dataDir: (str) directory where to save the CSV files
  @param outputFileName: (str) base name for the output file
  @param signalMean: (float) mean of the signal to generate
  @param signalPeriod: (float) period of the signal to generate
  @param sequenceLength: (int) sequence length of the signal to generate
  @param numPoints: (int) total number of points in the sequence
  @param signalAmplitude: (float) amplitude of the signal to generate
  @param numCategories: (int) number of categories labels
  @param noiseAmplitude: (float) amplitude of the white noise
  @return outFilePath: (str) path to the output file
  """
  fileName = "%s_%s" % (outputFileName, noiseAmplitude)

  if not dataDir:
    dataDir = DATA_DIR

  # make sure the directory exist. if not, create it.
  if not os.path.exists(dataDir):
    os.makedirs(dataDir)

  fileHandle = open("%s/%s.csv" % (dataDir, fileName), "wb")
  writer = csv.writer(fileHandle)
  writer.writerow(["x", "y", "label"])
  writer.writerow(["float", "float", "int"])
  writer.writerow(["", "", "C"])  # C is for category. 
  # WARNING: if the C flag is forgotten in the dataset, then all records will
  #  be arbitrarily put
  # in the same category (i.e. category 0). So make sure to have the C flag 
  # -- otherwise you'll get 100% classification accuracy regardless of 
  # the input data :-P


  endOfSequence = sequenceLength
  label = numCategories - 1
  for i in range(numPoints):

    noise = noiseAmplitude * random.random()

    if i == endOfSequence:
      endOfSequence += sequenceLength
      if label == 0:
        label = numCategories - 1
      else:
        label -= 1

    signal_modifier = 2 * (label + 1)
    x = signal_modifier * (i * math.pi) / signalPeriod
    m1 = signal_modifier * signalMean + signalAmplitude * math.sin(x) + noise

    writer.writerow([x, m1, label])

  fileHandle.close()

  outFilePath = os.path.join(dataDir,
                          "%s_%s.csv" % (outputFileName, noiseAmplitude))
  return outFilePath

if __name__ == "__main__":
  for whiteNoiseAmplitude in WHITE_NOISE_AMPLITUDES:
    generateData(DATA_DIR,
                 OUTFILE_NAME,
                 SIGNAL_MEAN,
                 SIGNAL_PERIOD,
                 SEQUENCE_LENGTH,
                 NUM_RECORDS,
                 SIGNAL_AMPLITUDE,
                 whiteNoiseAmplitude)
