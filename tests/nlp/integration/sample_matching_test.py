#!/usr/bin/env python
# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2015, Numenta, Inc.  Unless you have purchased from
# Numenta, Inc. a separate commercial license for this software code, the
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

import numpy
import os
import shutil
import unittest

from htmresearch.encoders import EncoderTypes
from htmresearch.frameworks.nlp.htm_runner import HTMRunner
from htmresearch.support.csv_helper import readCSV

import simplejson as json


DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_data")



class ClassificationModelsTest(unittest.TestCase):
  """Test class for ClassificationModelKeywords."""

  @staticmethod
  def runExperiment(runner):
    try:
      runner.setupData()
      runner.encodeSamples()
      runner.runExperiment()
    except Exception as e:
      print "Runner could not execute the experiment."
      raise e
    finally:
      # Cleanup
      shutil.rmtree(runner.model.modelDir.split("/")[0])


  def testHTM(self):
    """
    Tests ClassificationModelHTM in matching samples of text.
    
    Network model trains a KNNClassifier on all of the samples, then is
    queried with a new sample, returning a data structure (dict?) of the original samples
    and their distances to the query (similarity measure it total overlap).
    """
    runner = HTMRunner(dataPath=os.path.join(DATA_DIR, "responses.csv"),
                       resultsDir="",
                       experimentName="response_matching",
                       load=False,
                       modelName="ClassificationModelHTM",
                       modelModuleName="fluent.models.classify_htm",
                       numClasses=0,
                       plots=0,
                       orderedSplit=False,
                       trainSize=[35],
                       verbosity=0,
                       generateData=True,
                       classifierType="KNN")

    # setup data
    runner.setupData()
    
    # build model
    trial = 0
    runner.resetModel(trial)
    runner.training(trial)
    import pdb; pdb.set_trace()

    # query the model with sample text
    prototypeDistances = runner.model.queryModel("shower", False)
#    import pdb; pdb.set_trace()



    # remove  runner.dataFiles and runner.classificationFile

    # assert dict of len 35
    # assert min is __ w/ overlap __


if __name__ == "__main__":
     unittest.main()
