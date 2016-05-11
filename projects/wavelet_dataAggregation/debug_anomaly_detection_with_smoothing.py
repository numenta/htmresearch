# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2013-2015, Numenta, Inc.  Unless you have an agreement
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

from nab.corpus import DataFile
from nab.detectors.numenta.numenta_detector import NumentaDetector

import matplotlib.pyplot as plt
import numpy as np

plt.close('all')
plt.ion()


if __name__ == "__main__":
  srcPath = "debugFiles/machine_temperature_system_failure_thresh_0.02.csv"
  dataset1 = DataFile(srcPath)
  srcPath = "debugFiles/machine_temperature_system_failure_thresh_0.04.csv"
  dataset2 = DataFile(srcPath)

  # initialize both Numenta detectors with dataset1, to make sure the min/max
  # are the same and the detectors are identical
  detector1 = NumentaDetector(dataset1, .01)
  detector1.initialize()

  detector2 = NumentaDetector(dataset1, .01)
  detector2.initialize()


  rawScore1 = []
  rawScore2 = []
  encodingOverlap = []
  spOverlap = []
  for i in range(5):  # range(len(dataset1.data)):
    print
    print "Step {}".format(i)

    inputData = {"timestamp": dataset1.data["timestamp"][i],
                 "value": dataset1.data["value"][i], '_learning': False}
    (logScore, rawScore) = detector1.handleRecord(inputData)
    rawScore1.append(rawScore)


    inputData = {"timestamp": dataset2.data["timestamp"][i],
                 "value": dataset2.data["value"][i], '_learning': False}
    (logScore, rawScore) = detector2.handleRecord(inputData)
    rawScore2.append(rawScore)

    # Get the sensorRegion outputs
    sensorRegion1 = detector1.model._getSensorRegion().getSelf()
    timeEncoding1 = sensorRegion1.getOutputValues('sourceEncodings')[0]
    valueEncoding1 = sensorRegion1.getOutputValues('sourceEncodings')[1]

    sensorRegion2 = detector1.model._getSensorRegion().getSelf()
    timeEncoding2 = sensorRegion2.getOutputValues('sourceEncodings')[0]
    valueEncoding2 = sensorRegion2.getOutputValues('sourceEncodings')[1]

    # Make sure the encoder outputs are the same
    assert (np.sum(
      np.logical_and(timeEncoding1, timeEncoding2)) == np.sum(timeEncoding1))

    assert (np.sum(
      np.logical_and(valueEncoding1, valueEncoding2)) == np.sum(valueEncoding1))

    print "Encoder Output: "
    print np.concatenate((timeEncoding1, valueEncoding1)).nonzero()[0]

    encodingOverlap.append(np.sum(np.logical_and(valueEncoding1, valueEncoding2)))

    # Get SP inputs and outputs
    spRegion1 = detector1.model._getSPRegion().getSelf()
    spRegion2 = detector2.model._getSPRegion().getSelf()
    spInput1 = spRegion1._spatialPoolerInput.nonzero()[0]
    spInput2 = spRegion2._spatialPoolerInput.nonzero()[0]
    print "Detector 1, SP input"
    print spInput1
    print
    print "Detector 2, SP input"
    print spInput2
    print

    spOutput1 = spRegion1._spatialPoolerOutput
    spOutput2 = spRegion2._spatialPoolerOutput

    print "Detector 1, SP output"
    print spOutput1.nonzero()[0]
    print
    print "Detector 2, SP output"
    print spOutput2.nonzero()[0]
    print

    spOverlap.append(np.sum(np.logical_and(spOutput1, spOutput2)))

    print "timestamp {} value {} score1 {} score2 {}".format(
      dataset1.data["timestamp"][i],
      dataset1.data["value"][i],
      rawScore1[-1],
      rawScore2[-1]
    )

  fig, ax = plt.subplots(nrows=4, ncols=1)

  ax[0].plot(dataset1.data["value"])
  ax[0].plot(dataset2.data["value"])
  ax[0].set_xlim([0, len(dataset1.data)])
  ax[0].set_ylabel('Data Value')
  ax[0].legend(['data 1', 'data 2'])

  ax[1].plot(encodingOverlap)
  ax[1].set_ylabel("Encoder Overlap")

  ax[2].plot(spOverlap)
  ax[2].set_ylabel("SP output shared bits")

  ax[3].plot(rawScore1)
  ax[3].plot(rawScore2, 'r')
  ax[3].set_ylabel('Raw Anomaly Score')




  # sp1 = detector1.model._getSPRegion().getSelf()._sfdr
  # sp2 = detector1.model._getSPRegion().getSelf()._sfdr
  # spOutput1 = detector1.model._getSPRegion().getSelf()._spatialPoolerOutput
  # tm = detector1.model._getTPRegion().getSelf()._tfdr
  #
  # # compare sp permanences
  # maxDiff = np.zeros(sp1._numColumns)
  # for i in range(sp1._numColumns):
  #   maxDiff[i] = np.max(np.abs(sp1._permanences.getRow(i) - sp2._permanences.getRow(i)))