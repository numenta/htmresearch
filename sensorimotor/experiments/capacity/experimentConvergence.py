#!/usr/bin/env python
# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2014-2015, Numenta, Inc.  Unless you have an agreement
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

import experiment
import time


SHOW_PROGRESS_INTERVAL = 200000
IS_ONLINE = True
TRAINING_REPS = 10
PIECE_SIZE = 20


def run(numWorlds, numElems, params, outputDir, plotVerbosity,
        consoleVerbosity):
  # Setup params
  n = params["n"]
  w = params["w"]
  tmParams = params["tmParams"]
  tpParams = params["tpParams"]
  # isOnline = params["isOnline"]
  # onlineTrainingReps = params["onlineTrainingReps"] if isOnline else None
  completeSequenceLength = numElems ** 2
  print ("Experiment parameters: "
         "(# worlds = {0}, # elements = {1}, n = {2}, w = {3}, "
         "online = {4}, trainingReps = {5}, pieceSize = {6})".format(
         numWorlds, numElems, n, w, IS_ONLINE, TRAINING_REPS,
         PIECE_SIZE))
  print "Temporal memory parameters: {0}".format(tmParams)
  print "Temporal pooler parameters: {0}".format(tpParams)
  print

  # Setup experiment
  start = time.time()
  runner, exhaustiveAgents, randomAgents = experiment.setupExperiment(n, w,
                                                                      numElems,
                                                                      numWorlds,
                                                                      tmParams,
                                                                      tpParams)
  (sensorSequence,
   motorSequence,
   sensorimotorSequence,
   sequenceLabels) = runner.generateSequences(completeSequenceLength *
                                              TRAINING_REPS,
                                              exhaustiveAgents,
                                              verbosity=consoleVerbosity)

  # Convergence testing
  print "Testing: (worlds: {0}, elements: {1})...".format(numWorlds, numElems)
  print

  pieces = len(sensorSequence) / PIECE_SIZE
  print "Time\tStability Mean\tMax\tStdDev"
  for i in xrange(pieces - 1):
    beginIdx = i * PIECE_SIZE
    endIdx = (i + 1) * PIECE_SIZE
    seqPiece = (sensorSequence[beginIdx: endIdx],
                motorSequence[beginIdx: endIdx],
                sensorimotorSequence[beginIdx: endIdx],
                sequenceLabels[beginIdx: endIdx])

    if IS_ONLINE:
      runner.feedLayers(seqPiece, tmLearn=True, tpLearn=True,
                        verbosity=consoleVerbosity,
                        showProgressInterval=SHOW_PROGRESS_INTERVAL)
    else:
      runner.feedLayers(seqPiece, tmLearn=True, tpLearn=False,
                        verbosity=consoleVerbosity,
                        showProgressInterval=SHOW_PROGRESS_INTERVAL)
      runner.feedLayers(seqPiece, tmLearn=False, tpLearn=True,
                        verbosity=consoleVerbosity,
                        showProgressInterval=SHOW_PROGRESS_INTERVAL)

    # This method call resets the monitor mixins' histories
    experiment.runTestPhase(runner, randomAgents, numWorlds, numElems,
                            completeSequenceLength, consoleVerbosity)
    metric = runner.tp.mmGetMetricStabilityConfusion()
    print "{0}\t{1}\t{2}\t{3}".format(endIdx, round(metric.mean, 3),
                                      metric.max, metric.standardDeviation)
  print

  elapsed = int(time.time() - start)
  print "Total time: {0:2} seconds.".format(elapsed)

  # Write results to output file
  experiment.writeOutput(outputDir, runner, numElems, numWorlds, elapsed)
  # if plotVerbosity >= 1:
  #   raw_input("Press any key to exit...")



if __name__ == "__main__":
  (options, args, params) = experiment._getArgs()
  run(int(args[0]), int(args[1]), params, args[3], options.plotVerbosity,
      options.consoleVerbosity)
