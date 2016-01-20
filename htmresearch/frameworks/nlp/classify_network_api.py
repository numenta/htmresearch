# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2016, Numenta, Inc.  Unless you have purchased from
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

import os
import numpy

from tabulate import tabulate

from nupic.engine import Network

from htmresearch.frameworks.classification.classification_network import (
  configureNetwork)
from htmresearch.encoders import EncoderTypes
from htmresearch.encoders.cio_encoder import CioEncoder
from htmresearch.frameworks.nlp.classification_model import ClassificationModel

class ClassificationNetworkAPI(ClassificationModel):
  """
  Helper subclass of ClassificationModel for those models that use the network
  API. This class will contain much of the repeated code.

  This class assumes subclasses will instantiate a model using the Network API.
  It will be a model containing a LanguageSensor and a KNNCLassifierRegion.
  """

  def __init__(self,
               networkConfig=None,
               **kwargs):
    """
    @param networkConfig      (dict)    Network configuration dict with region
                                        parameters.

    Note classifierMetric is not specified here as it is in other models. This
    is done in the network config file.
    """
    super(ClassificationNetworkAPI, self).__init__(**kwargs)

    self.networkConfig = networkConfig
    self.currentDocument = None
    self.network = None

    if self.networkConfig is None:
      raise RuntimeError("Must specify a network config dict")


  def getClassifier(self):
    """
    Returns the classifier for the model.
    """
    return self.classifierRegion.getSelf().getAlgorithmInstance()


  def _initializeRegionHelpers(self):
    """
    Set helper member variables once network has been initialized. This
    should be called after self.network has been initialized. This method
    will also be called from _deSerializeExtraData()
    """
    learningRegions = []
    for region in self.network.regions.values():
      spec = region.getSpec()
      if spec.parameters.contains("learningMode"):
        learningRegions.append(region)

    # Always a sensor and classifier region.
    self.sensorRegion = self.network.regions[
      self.networkConfig["sensorRegionConfig"].get("regionName")]
    self.classifierRegion = self.network.regions[
      self.networkConfig["classifierRegionConfig"].get("regionName")]

    # There is sometimes a TP region
    self.tpRegion = None
    if self.networkConfig.has_key("tpRegionConfig"):
      self.tpRegion = self.network.regions[
        self.networkConfig["tpRegionConfig"].get("regionName")]

    # There is sometimes a TM region
    self.tmRegion = None
    if self.networkConfig.has_key("tmRegionConfig"):
      self.tmRegion = self.network.regions[
        self.networkConfig["tmRegionConfig"].get("regionName")]

    self.learningRegions = learningRegions

    self.network.enableProfiling()


  def reset(self):
    """
    Issue a reset signal to the model. The assumption is that a sequence has
    just ended and a new sequence is about to begin.
    """
    # TODO: Introduce a consistent reset method name in Regions
    for r in self.learningRegions:
      if r.type in ["py.TemporalPoolerRegion", "py.TMRegion"]:
        r.executeCommand(["reset"])
      elif r.type == "py.TPRegion":
        r.executeCommand(["resetSequenceStates"])


  def printRegionOutputs(self):
    """
    Print the outputs of regions to console for debugging, depending on
    verbosity level.
    """
    print "================== HTM Debugging output:"

    # Basic output
    print "Sensor number of nonzero bits:",
    print len(self.sensorRegion.getOutputData("dataOut").nonzero()[0])
    if self.tmRegion is not None:
      print "TM region number of activeCells: ",
      print len(self.tmRegion.getOutputData("bottomUpOut").nonzero()[0]),
      print "Number of predictedActiveCells: ",
      print len(self.tmRegion.getOutputData("predictedActiveCells").nonzero()[0])

    print "Classifier categoriesOut: ",
    print self.classifierRegion.getOutputData("categoriesOut")[0:self.numLabels]
    print "Classifier categoryProbabilitiesOut",
    print self.classifierRegion.getOutputData("categoryProbabilitiesOut")[0:self.numLabels]

    # Really detailed output
    if self.verbosity >= 3:
      print "Sensor output:",
      print self.sensorRegion.getOutputData("dataOut").nonzero()
      print "Sensor categoryOut:",
      print self.sensorRegion.getOutputData("categoryOut")

      if self.tmRegion is not None:
        print "TM region bottomUpOut: ",
        print self.tmRegion.getOutputData("bottomUpOut").nonzero()
        print "TM region predictedActiveCells: ",
        print self.tmRegion.getOutputData("predictedActiveCells").nonzero()

      print "Classifier bottomUpIn: ",
      print self.classifierRegion.getInputData("bottomUpIn").nonzero()
      print "Classifier categoryIn: ",
      print self.classifierRegion.getInputData("categoryIn")[0:self.numLabels]



  def dumpProfile(self):
    """
    Print region profiling information in a nice format.
    """
    print "Profiling information for {}".format(type(self).__name__)
    totalTime = 0.000001
    for region in self.network.regions.values():
      timer = region.computeTimer
      totalTime += timer.getElapsed()

    count = 1
    profileInfo = []
    for region in self.network.regions.values():
      timer = region.computeTimer
      count = max(timer.getStartCount(), count)
      profileInfo.append([region.name,
                          timer.getStartCount(),
                          timer.getElapsed(),
                          100.0*timer.getElapsed()/totalTime,
                          timer.getElapsed()/max(timer.getStartCount(),1)])

    profileInfo.append(["Total time", "", totalTime, "100.0", totalTime/count])
    print tabulate(profileInfo, headers=["Region", "Count",
                   "Elapsed", "Pct of total", "Secs/iteration"],
                   tablefmt = "grid", floatfmt="6.3f")

    if self.tmRegion is not None:
      if self.tmRegion.getSpec().commands.contains("prettyPrintTraces"):
        self.tmRegion.executeCommand(["prettyPrintTraces"])


  def __getstate__(self):
    """
    Return serializable state.  This function will return a version of the
    __dict__ with data that shouldn't be pickled stripped out. For example,
    Network API instances are stripped out because they have their own
    serialization mechanism.

    See also: _serializeExtraData()
    """
    state = self.__dict__.copy()
    # Remove member variables that we can't pickle
    state.pop("network")
    state.pop("sensorRegion")
    state.pop("classifierRegion")
    state.pop("tpRegion")
    state.pop("learningRegions")
    state.pop("tmRegion")

    return state


  def _serializeExtraData(self, extraDataDir):
    """
    Protected method that is called during serialization with an external
    directory path. We override it here to save the Network API instance.

    @param extraDataDir (string) Model's extra data directory path
    """
    self.network.save(os.path.join(extraDataDir, "network.nta"))


  def _deSerializeExtraData(self, extraDataDir):
    """
    Protected method that is called during deserialization (after __setstate__)
    with an external directory path. We override it here to load the Network API
    instance.

    @param extraDataDir (string) Model's extra data directory path
    """
    self.network = Network(os.path.join(extraDataDir, "network.nta"))
    self._initializeRegionHelpers()
