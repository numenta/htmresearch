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
import csv
import os

from classification_network import createNetwork
from classify_sensor_data import runNetwork
from fluent.encoders.cio_encoder import CioEncoder
from nupic.data.file_record_stream import FileRecordStream


## TODOs:
#   - replace outFile writing w/ csv writer - commented out

_INPUT_FILE_PATH = os.path.abspath("../../nupic.fluent/data/sample_reviews_multi/sample_reviews_data_training_network2.csv")
# _INPUT_FILE_PATH = os.path.abspath("../../nupic.fluent/network_experiment/data.csv")
_OUTPUT_FILE_NAME = ""

TOTAL_NUMBER_OF_CATEGORIES = 4



def run(outFile):

  dataSource = FileRecordStream(streamID=_INPUT_FILE_PATH)
  numRecords = dataSource.getDataRowCount()
  # Partition records into training sets for SP, TM, and classifier
  partitions = [numRecords/4, numRecords/2, numRecords*3/4]

  print "Creating network"
  encoder = CioEncoder(cacheDir="./fluent_experiments/cache")
  network = createNetwork((dataSource, "py.LanguageSensor", encoder, TOTAL_NUMBER_OF_CATEGORIES))
  # outputPath = os.path.join(os.path.dirname(__file__), _OUTPUT_FILE_NAME)
  network.initialize()

  # with open(outputPath, "w") as outputFile:
    # writer = csv.writer(outputFile)
  print "Running network"
    # print "Writing output to: {}".format(outputPath)
  numCorrect, numTestRecords, predictionAccuracy = runNetwork(
      network, numRecords, partitions, outFile)
  print "NLP network run finished"


if __name__ == "__main__":
  outFile = open("results/network.out", 'wb')
  run(outFile)
  outFile.close()
