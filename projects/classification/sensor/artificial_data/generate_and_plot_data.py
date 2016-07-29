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

from htmresearch.frameworks.classification.utils.sensor_data import (
  generateSensorData)
from htmresearch.frameworks.classification.utils.sensor_data import (
  plotSensorData)

from settings import (OUTFILE_NAME,
                      SEQUENCE_LENGTH,
                      NUM_CATEGORIES,
                      NUM_RECORDS,
                      WHITE_NOISE_AMPLITUDES,
                      SIGNAL_AMPLITUDES,
                      SIGNAL_MEANS,
                      SIGNAL_PERIODS,
                      RESULTS_DIR,
                      MODEL_PARAMS_DIR,
                      DATA_DIR)



def _generateData():
  """
  Generate CSV data to plot.
  @return outFiles: (list) paths to output files
  """
  outFiles = []
  for noiseAmplitude in WHITE_NOISE_AMPLITUDES:
    for signalMean in SIGNAL_MEANS:
      for signalAmplitude in SIGNAL_AMPLITUDES:
        for signalPeriod in SIGNAL_PERIODS:
          outFile = generateSensorData(DATA_DIR,
                                       OUTFILE_NAME,
                                       signalMean,
                                       signalPeriod,
                                       SEQUENCE_LENGTH,
                                       NUM_RECORDS,
                                       signalAmplitude,
                                       NUM_CATEGORIES,
                                       noiseAmplitude)
          outFiles.append(outFile)

  return outFiles



def main():
  csvFiles = _generateData()
  plotSensorData(csvFiles, SEQUENCE_LENGTH)



if __name__ == "__main__":
  main()
