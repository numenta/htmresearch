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
  generateSensorData, plotSensorData)

from settings import (SIGNAL_TYPES,
                      NUM_PHASES,
                      NUM_REPS,
                      NUM_CATEGORIES,
                      WHITE_NOISE_AMPLITUDES,
                      SIGNAL_AMPLITUDES,
                      SIGNAL_MEANS,
                      DATA_DIR,
                      NOISE_LENGTHS)



def _generateExpData():
  """
  Generate CSV data to plot.
  @return expSetups: (list of dict) info about each experiment setup.
  """

  expSetups = []
  for signalType in SIGNAL_TYPES:
    for noiseAmplitude in WHITE_NOISE_AMPLITUDES:
      for signalMean in SIGNAL_MEANS:
        for signalAmplitude in SIGNAL_AMPLITUDES:
          for numCategories in NUM_CATEGORIES:
            for numReps in NUM_REPS:
              for numPhases in NUM_PHASES:
                for noiseLength in NOISE_LENGTHS:
                  expSetup = generateSensorData(signalType,
                                                DATA_DIR,
                                                numPhases,
                                                numReps,
                                                signalMean,
                                                signalAmplitude,
                                                numCategories,
                                                noiseAmplitude,
                                                noiseLength)

                  expSetups.append(expSetup)

  return expSetups



def main():
  expSetups = _generateExpData()
  plotSensorData(expSetups)



if __name__ == "__main__":
  main()
