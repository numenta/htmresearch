import pprint
import time

import numpy as np


from nupic.algorithms.KNNClassifier import KNNClassifier
from nupic.bindings.math import GetNTAReal
from nupic.research.monitor_mixin.monitor_mixin_base import MonitorMixinBase
from nupic.research.monitor_mixin.temporal_memory_monitor_mixin import (
    TemporalMemoryMonitorMixin)

from htmresearch.algorithms.general_temporal_memory import (
     GeneralTemporalMemory)

from htmresearch.algorithms.spatiotemporal_pooler import SpatiotemporalPooler
from nupic.research.monitor_mixin.monitor_mixin_base import MonitorMixinBase

realDType = GetNTAReal()



class SpatiotemporalPoolerMixin(MonitorMixinBase):
  pass



class MontitoredSpatiotemporalPooler(SpatiotemporalPoolerMixin, SpatiotemporalPooler):
  pass



class SpatiotemporalPoolerExperiment(object):
  """
  This class defines a Spatiotemporal Pooler network and provides methods
  to run the network on data sequences.
  """


  DEFAULT_TEMPORAL_MEMORY_PARAMS = {"columnDimensions": (1024,),
                                    "cellsPerColumn": 8,
                                    "activationThreshold": 20,
                                    "initialPermanence": 0.5,
                                    "connectedPermanence": 0.6,
                                    "minThreshold": 20,
                                    "maxNewSynapseCount": 30,
                                    "permanenceIncrement": 0.10,
                                    "permanenceDecrement": 0.02,
                                    "seed": 42,
                                    "learnOnOneCell": False}


  DEFAULT_SPATIOTEMPORAL_POOLER_PARAMS = {# Spatial Pooler Params
                                 # inputDimensions set to TM cell count
                                 # potentialRadius set to TM cell count
                                 "inputDimensions": (1024,),
                                 # "columnDimensions": [1024],
                                 "columnDimensions": (1024,),
                                 "numActiveColumnsPerInhArea": 10,
                                 # "numActiveColumnsPerInhArea": 3,
                                 "stimulusThreshold": 0,
                                 "synPermInactiveDec": 0.01,
                                 "synPermActiveInc": 0.1,
                                 "synPermConnected": 0.1,
                                 # "potentialPct": 0.5,
                                 "potentialPct": .5,
                                 "globalInhibition": True,
                                 "localAreaDensity": -1,
                                 "minPctOverlapDutyCycle": 0.001,
                                 "minPctActiveDutyCycle": 0.001,
                                 "dutyCyclePeriod": 1000,
                                 # "maxBoost": 10.0,
                                 "maxBoost": 1.0,
                                 "seed": 42,
                                 "spVerbosity": 0,
                                 "wrapAround": True,

                                 # Spatiotemporal Pooler Params
                                 "historyLength": 10,
                                 # "activeOverlapWeight": 1.0,
                                 # "predictedActiveOverlapWeight": 10.0,
                                 # "maxUnionActivity": 0.20,
                                 # "exciteFunctionType": 'Fixed',
                                 # "decayFunctionType": 'NoDecay'
                               }

  DEFAULT_CLASSIFIER_PARAMS = {"k": 1,
                               "distanceMethod": "rawOverlap",
                               "distThreshold": 0}



  def __init__(self, tmOverrides=None, stpOverrides=None,
               classifierOverrides=None, seed=42, consoleVerbosity=0):
    # print "Initializing Temporal Memory..."
    # params = dict(self.DEFAULT_TEMPORAL_MEMORY_PARAMS)
    # params.update(tmOverrides or {})
    # params["seed"] = seed
    # self.tm = MonitoredFastGeneralTemporalMemory(mmName="TM", **params)

    print "Initializing Spatiotemporal Pooler..."
    start = time.time()
    params = dict(self.DEFAULT_SPATIOTEMPORAL_POOLER_PARAMS)
    params.update(stpOverrides or {})
    # params["inputDimensions"] = [self.tm.numberOfCells()]
    # params["potentialRadius"] = self.tm.numberOfCells()
    params["seed"] = seed
    self.params = params
    # self.up = MonitoredUnionTemporalPooler(mmName="UP", **params)
    self.stp = MontitoredSpatiotemporalPooler(**params)
    elapsed = int(time.time() - start)
    print "Total time: {0:2} seconds.".format(elapsed)

    # print "Initializing KNN Classifier..."
    # params = dict(self.DEFAULT_CLASSIFIER_PARAMS)
    # params.update(classifierOverrides or {})
    # self.classifier = KNNClassifier(**params)


  def runNetworkOnSequences(self, inputSequences, inputCategories, tmLearn=True,
                            stpLearn=None, classifierLearn=False, verbosity=0,
                            progressInterval=None):
    """
    Runs Union Temporal Pooler network on specified sequence.

    @param inputSequences           One or more sequences of input patterns.
                                    Each should be terminated with None.

    @param inputCategories          A sequence of category representations
                                    for each element in inputSequences
                                    Each should be terminated with None.

    @param tmLearn:   (bool)        Temporal Memory learning mode
    @param upLearn:   (None, bool)  Union Temporal Pooler learning mode. If None,
                                    Union Temporal Pooler will not be run.
    @param classifierLearn: (bool)  Classifier learning mode

    @param progressInterval: (int)  Interval of console progress updates
                                    in terms of timesteps.
    """

    currentTime = time.time()

    # columnActivations = list()
    columnActivations = np.ndarray((len(inputSequences), self.stp._numColumns))
    
    unionedInput = np.ndarray((len(inputSequences), self.stp._numInputs))

    for i in xrange(len(inputSequences)):
      sensorPattern = inputSequences[i]
      inputCategory = inputCategories[i]
            
      columnActivations[i,:] = self.runNetworkOnPattern(sensorPattern,
                                 tmLearn=tmLearn,
                                 stpLearn=stpLearn,
                                 sequenceLabel=inputCategory,
                                 unionedInputMonitor=unionedInput[i])

      # if classifierLearn and sensorPattern is not None:
      #   unionSDR = self.up.getUnionSDR()
      #   upCellCount = self.up.getColumnDimensions()
      #   self.classifier.learn(unionSDR, inputCategory, isSparse=upCellCount)
      #   if verbosity > 1:
      #     pprint.pprint("{0} is category {1}".format(unionSDR, inputCategory))

      if progressInterval is not None and i > 0 and i % progressInterval == 0:
        elapsed = (time.time() - currentTime) / 60.0
        print ("Ran {0} / {1} elements of sequence in "
               "{2:0.2f} minutes.".format(i, len(inputSequences), elapsed))
        currentTime = time.time()
        # print MonitorMixinBase.mmPrettyPrintMetrics(
        #   self.tm.mmGetDefaultMetrics())
    #
    # if verbosity >= 2:
    #   traces = self.tm.mmGetDefaultTraces(verbosity=verbosity)
    #   print MonitorMixinBase.mmPrettyPrintTraces(traces,
    #                                              breakOnResets=
    #                                              self.tm.mmGetTraceResets())
    #
    #   if upLearn is not None:
    #     traces = self.up.mmGetDefaultTraces(verbosity=verbosity)
    #     print MonitorMixinBase.mmPrettyPrintTraces(traces,
    #                                                breakOnResets=
    #                                                self.up.mmGetTraceResets())
      print
    
    # print columnActivations
    
    # import pprint
    # pprint.pprint(columnActivations)
    
    return columnActivations, unionedInput

  def runNetworkOnPattern(self, sensorPattern, tmLearn=True, stpLearn=None,
                          sequenceLabel=None, unionedInputMonitor=None):
    if sensorPattern is None:
      # self.tm.reset()
      self.stp.reset()
    else:
      # self.tm.compute(sensorPattern,
      #                 formInternalConnections=True,
      #                 learn=tmLearn,
      #                 sequenceLabel=sequenceLabel)

      # if stpLearn is not None:
        # activeCells, predActiveCells, burstingCols, = self.getUnionTemporalPoolerInput()
        
      activeCells = np.zeros(self.params["inputDimensions"])
      print np.array(list(sensorPattern), dtype=int)
      activeCells[np.array(list(sensorPattern), dtype=int)] = 1
      
      predActiveCells = activeCells.copy()
      
      activeColumns = self.stp.compute(activeCells,
                                       predActiveCells,
                                       learn=stpLearn,
                                       unionedInputMonitor=unionedInputMonitor)

      return activeColumns


  # def getSpatiotemporalPoolerInput(self):
    
    

  # def getUnionTemporalPoolerInput(self):
  #   """
  #   Gets the Union Temporal Pooler input from the Temporal Memory
  #   """
  #   activeCells = numpy.zeros(self.tm.numberOfCells()).astype(realDType)
  #   activeCells[list(self.tm.activeCellsIndices())] = 1
  #
  #   predictedActiveCells = numpy.zeros(self.tm.numberOfCells()).astype(
  #     realDType)
  #   predictedActiveCells[list(self.tm.predictedActiveCellsIndices())] = 1
  #
  #   burstingColumns = numpy.zeros(self.tm.numberOfColumns()).astype(realDType)
  #   burstingColumns[list(self.tm.unpredictedActiveColumns)] = 1
  #
  #   return activeCells, predictedActiveCells, burstingColumns


  # def getBurstingColumnsStats(self):
  #   """
  #   Gets statistics on the Temporal Memory's bursting columns. Used as a metric
  #   of Temporal Memory's learning performance.
  #   :return: mean, standard deviation, and max of Temporal Memory's bursting
  #   columns over time
  #   """
  #   traceData = self.tm.mmGetTraceUnpredictedActiveColumns().data
  #   resetData = self.tm.mmGetTraceResets().data
  #   countTrace = []
  #   for x in xrange(len(traceData)):
  #     if not resetData[x]:
  #       countTrace.append(len(traceData[x]))
  #
  #   mean = numpy.mean(countTrace)
  #   stdDev = numpy.std(countTrace)
  #   maximum = max(countTrace)
  #   return mean, stdDev, maximum
