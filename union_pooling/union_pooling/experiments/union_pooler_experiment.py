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

from union_pooling.union_pooler import UnionPooler

# TODO Potential Mixin classes go here

class UnionPoolerExperiment(object):


  DEFAULT_UNION_POOLER_PARAMS = {"synPermInactiveDec": 0,
                                 "synPermActiveInc": 0.001,
                                 "synPredictedInc": 0.5,
                                 "potentialPct": 0.9,
                                 "initConnectedPct": 0.5,

                                 # Experimental Parameters require client
                                 # override
                                 "numActiveColumnsPerInhArea": "Sorry"}


  def __init__(self, tmOverrides=None, tpOverrides=None, seed=42):
    # # Initialize Layer 4 temporal memory
    # params = dict(self.DEFAULT_TM_PARAMS)
    # params.update(tmOverrides or {})
    # params["seed"] = seed
    # self._checkParams(params)
    # self.tm = MonitoredGeneralTemporalMemory(mmName="TM", **params)
    #
    # Initialize Union Pooler layer
    params = dict(self.DEFAULT_UNION_POOLER_PARAMS)
    # params["inputDimensions"] = [self.tm.numberOfCells()]
    # params["potentialRadius"] = self.tm.numberOfCells()
    # params["seed"] = seed
    # params.update(tpOverrides or {})
    # self._checkParams(params)
    # self.unionPooler = MonitoredTemporalPooler(mmName="TP", **params)
    self.unionPooler = UnionPooler(**params)


  def getUnionPoolerInput(self):
    pass

    # """
    # Given an instance of the TM, format the information we need to send to the
    # TP.
    # """
    # # all currently active cells in layer 4
    # tpInputVector = numpy.zeros(
    #               self.tm.numberOfCells()).astype(realDType)
    # tpInputVector[list(self.tm.activeCellsIndices())] = 1
    #
    # # bursting columns in layer 4
    # burstingColumns = numpy.zeros(
    #   self.tm.numberOfColumns()).astype(realDType)
    # burstingColumns[list(self.tm.unpredictedActiveColumns)] = 1
    #
    # # correctly predicted cells in layer 4
    # correctlyPredictedCells = numpy.zeros(
    #   self.tm.numberOfCells()).astype(realDType)
    # correctlyPredictedCells[list(self.tm.predictedActiveCellsIndices())] = 1
    #
    # return tpInputVector, burstingColumns, correctlyPredictedCells



  def feedTransition(self, sensorPattern, motorPattern, sensorimotorPattern,
                     tmLearn=True, tpLearn=None, sequenceLabel=None):
    pass
    # if sensorPattern is None:
    #   self.tm.reset()
    #   self.tp.reset()
    #
    # else:
    #   # Feed the TM
    #   self.tm.compute(sensorPattern,
    #             activeExternalCells=motorPattern,
    #             formInternalConnections=True,
    #             learn=tmLearn,
    #             sequenceLabel=sequenceLabel)
    #
    #   # If requested, feed the TP
    #   if tpLearn is not None:
    #     tpInputVector, burstingColumns, correctlyPredictedCells = (
    #         self.formatInputForTP())
    #     activeArray = numpy.zeros(self.tp.getNumColumns())
    #
    #     self.tp.compute(tpInputVector,
    #                     tpLearn,
    #                     activeArray,
    #                     burstingColumns,
    #                     correctlyPredictedCells,
    #                     sequenceLabel=sequenceLabel)


  def feedLayers(self, sequences, tmLearn=True, tpLearn=None, verbosity=0,
                 showProgressInterval=None):
    pass
    # """
    # Feed the given sequences to the HTM algorithms.
    #
    # @param tmLearn:   (bool)      Either False, or True
    # @param tpLearn:   (None,bool) Either None, False, or True. If None,
    #                               temporal pooler will be skipped.
    #
    # @param showProgressInterval: (int) Prints progress every N iterations,
    #                                    where N is the value of this param
    # """
    # (sensorSequence,
    #  motorSequence,
    #  sensorimotorSequence,
    #  sequenceLabels) = sequences
    #
    # currentTime = time.time()
    #
    # for i in xrange(len(sensorSequence)):
    #   sensorPattern = sensorSequence[i]
    #   motorPattern = motorSequence[i]
    #   sensorimotorPattern = sensorimotorSequence[i]
    #   sequenceLabel = sequenceLabels[i]
    #
    #   self.feedTransition(sensorPattern, motorPattern, sensorimotorPattern,
    #                       tmLearn=tmLearn, tpLearn=tpLearn,
    #                       sequenceLabel=sequenceLabel)
    #
    #   if (showProgressInterval is not None and
    #       i > 0 and
    #       i % showProgressInterval == 0):
    #     print ("Fed {0} / {1} elements of the sequence "
    #            "in {2:0.2f} seconds.".format(
    #              i, len(sensorSequence), time.time() - currentTime))
    #     currentTime = time.time()
    #
    # if verbosity >= 2:
    #   # Print default TM traces
    #   traces = self.tm.mmGetDefaultTraces(verbosity=verbosity)
    #   print MonitorMixinBase.mmPrettyPrintTraces(traces,
    #                                              breakOnResets=
    #                                              self.tm.mmGetTraceResets())
    #
    #   if tpLearn is not None:
    #     # Print default TP traces
    #     traces = self.tp.mmGetDefaultTraces(verbosity=verbosity)
    #     print MonitorMixinBase.mmPrettyPrintTraces(traces,
    #                                                breakOnResets=
    #                                                self.tp.mmGetTraceResets())
    #   print
