import csv
import os
import sys
import time
import yaml
from optparse import OptionParser

from nupic.research.monitor_mixin.monitor_mixin_base import MonitorMixinBase

from experiments.capacity import experiment
from sensorimotor.exhaustive_one_d_agent import ExhaustiveOneDAgent
from sensorimotor.one_d_world import OneDWorld
from sensorimotor.one_d_universe import OneDUniverse
from sensorimotor.random_one_d_agent import RandomOneDAgent
from sensorimotor.sensorimotor_experiment_runner import (
    SensorimotorExperimentRunner)

RANDOM_SEED = 42
SHOW_PROGRESS_INTERVAL = 200
NUM_TEST_SEQUENCES = 4



def setupExperiment(n, w, numWorldSequences, worldSequenceLength, numElements):
  """
  Returns a list of world sequences. Each sequence is broken up into a
  training section and a test section, both sequences of corresponding worlds

  [ [[trainAgents0], [testAgents0]],
    [[trainAgents1], [testAgents1]],
    ...
    [[trainAgentsN], [testAgentsN]]  ]
  """
  print "Setting up experiment..."
  universe = OneDUniverse(nSensor=n, wSensor=w,
                          nMotor=n, wMotor=w)

  worldSequences = []
  for sequence in xrange(numWorldSequences):

    exhaustiveAgents = []
    randomAgents = []
    for world in xrange(worldSequenceLength):
      i = sequence * worldSequenceLength + world
      elements = range(i * numElements, (i + 1) * numElements)
      agent = ExhaustiveOneDAgent(OneDWorld(universe, elements), 0)
      exhaustiveAgents.append(agent)

      possibleMotorValues = range(-numElements, numElements + 1)
      possibleMotorValues.remove(0)
      agent = RandomOneDAgent(OneDWorld(universe, elements), numElements / 2,
                              possibleMotorValues=possibleMotorValues,
                              seed=RANDOM_SEED)
      randomAgents.append(agent)

    worldSequences.append([exhaustiveAgents, randomAgents])

  print "Done setting up experiment."
  print
  return worldSequences



def trainWorldSequences(runner, completeSequenceLength, onlineTrainingReps,
                        worldSequences, consoleVerbosity):
  for sequence in worldSequences:
    trainAgents = sequence[0]
    sequences = runner.generateSequences(completeSequenceLength, trainAgents,
                                         numSequences=onlineTrainingReps,
                                         verbosity=consoleVerbosity)
    runner.feedLayers(sequences, tmLearn=True, tpLearn=True,
                      verbosity=consoleVerbosity,
                      showProgressInterval=SHOW_PROGRESS_INTERVAL)



def trainWorlds(runner, completeSequenceLength, onlineTrainingReps,
                worldSequences, consoleVerbosity):
  for sequence in worldSequences:
    trainAgents = sequence[0]
    sequences = runner.generateSequences(completeSequenceLength *
                                         onlineTrainingReps, trainAgents,
                                         verbosity=consoleVerbosity)
    runner.feedLayers(sequences, tmLearn=True, tpLearn=True,
                      verbosity=consoleVerbosity,
                      showProgressInterval=SHOW_PROGRESS_INTERVAL)



def testPhase(runner, worldSequences, completeSequenceLength,
              consoleVerbosity):

  runner.tm.mmClearHistory()
  runner.tp.mmClearHistory()

  for sequence in worldSequences:
    testAgents = sequence[1]
    sequences = runner.generateSequences(completeSequenceLength /
                                         NUM_TEST_SEQUENCES,
                                         testAgents, verbosity=consoleVerbosity,
                                         numSequences=NUM_TEST_SEQUENCES)
    runner.feedLayers(sequences, tmLearn=False, tpLearn=False,
                      verbosity=consoleVerbosity,
                      showProgressInterval=SHOW_PROGRESS_INTERVAL)



def writeOutput(outputDir, runner, numWorldSequences, worldSequenceLength,
                numElements, elapsedTime):
  if not os.path.exists(outputDir):
    os.makedirs(outputDir)
  fileName = "{0:0>3}x{1:0>3}x{2:0>3}.csv".format(numWorldSequences,
                                                  worldSequenceLength,
                                                  numElements)
  filePath = os.path.join(outputDir, fileName)
  with open(filePath, "wb") as outputFile:
    csvWriter = csv.writer(outputFile)
    header = ["# world sequences", "sequence length", "# elements", "duration"]
    row = [numWorldSequences, worldSequenceLength, numElements, elapsedTime]
    for metric in (runner.tp.mmGetDefaultMetrics() +
                   runner.tm.mmGetDefaultMetrics()):
      header += ["{0} ({1})".format(metric.prettyPrintTitle(), x) for x in
                ["min", "max", "sum", "mean", "stddev"]]
      row += [metric.min, metric.max, metric.sum, metric.mean,
              metric.standardDeviation]
    csvWriter.writerow(header)
    csvWriter.writerow(row)
    outputFile.flush()



def run(numWorldSequences, worldSequenceLength, numElements, params, outputDir,
        plotVerbosity, consoleVerbosity):
  # Setup params
  n = params["n"]
  w = params["w"]
  tmParams = params["tmParams"]
  tpParams = params["tpParams"]
  isOnline = params["isOnline"]
  onlineTrainingReps = params["onlineTrainingReps"] if isOnline else None
  isTrainWorlds = params["isTrainWorlds"]
  completeSequenceLength = numElements ** 2
  print ("Experiment parameters: "
         "(# worldSequences = {0}, # worldSequenceLength = {1}, # elements = {"
         "2}, n = {3}, w = {4}, online = {5}, onlineReps = {6})".format(
         numWorldSequences, worldSequenceLength, numElements, n, w, isOnline,
         onlineTrainingReps))
  print "Temporal memory parameters: {0}".format(tmParams)
  print "Temporal pooler parameters: {0}".format(tpParams)
  print

  # Setup experiment
  start = time.time()
  runner = SensorimotorExperimentRunner(tmOverrides=tmParams,
                                        tpOverrides=tpParams,
                                        seed=RANDOM_SEED)
  worldSequences = setupExperiment(n, w, numWorldSequences,
                                   worldSequenceLength, numElements)

  if isTrainWorlds:
    # Train on each world
    trainWorlds(runner, completeSequenceLength, onlineTrainingReps,
                worldSequences, consoleVerbosity)
  else:
    # Train on world sequences
    trainWorldSequences(runner, completeSequenceLength, onlineTrainingReps,
                        worldSequences, consoleVerbosity)

  print MonitorMixinBase.mmPrettyPrintMetrics(runner.tp.mmGetDefaultMetrics() +
                                              runner.tm.mmGetDefaultMetrics())
  print
  if plotVerbosity > 0:
    experiment.plotExperimentState(runner, plotVerbosity, numWorldSequences *
                                   worldSequenceLength, numElements, isOnline,
                                   "Training")

  print "Testing (worldSequences: {0}, worldSequenceLength: {1}, elements: {" \
        "2})...".format(numWorldSequences, worldSequenceLength, numElements)
  testPhase(runner, worldSequences, completeSequenceLength, consoleVerbosity)
  print MonitorMixinBase.mmPrettyPrintMetrics(runner.tp.mmGetDefaultMetrics() +
                                              runner.tm.mmGetDefaultMetrics())
  print
  if plotVerbosity > 0:
    experiment.plotExperimentState(runner, plotVerbosity, numWorldSequences *
                                   worldSequenceLength, numElements, isOnline,
                                   "Training")

  elapsed = int(time.time() - start)
  print "Total time: {0:2} seconds.".format(elapsed)

  # Write results to output file
  writeOutput(outputDir, runner, numWorldSequences, worldSequenceLength,
              numElements, elapsed)

  if plotVerbosity > 0:
    raw_input("Press any key to exit...")



def _getArgs():
  parser = OptionParser(usage="%prog WORLD_SEQUENCES WORLD_SEQUENCE_LENGTH "
                              "NUM_ELEMENTS PARAMS_DIR OUTPUT_DIR [options]"
                              "\n\nRun feedback experiment with specified "
                              "worlds and elements using params in PARAMS_DIR "
                              "and outputting results to OUTPUT_DIR.")
  parser.add_option("-p",
                    "--plot",
                    type=int,
                    default=0,
                    dest="plotVerbosity",
                    help="Plotting verbosity: 0 => none, 1 => summary plots, "
                         "2 => detailed plots")
  parser.add_option("-c",
                    "--console",
                    type=int,
                    default=0,
                    dest="consoleVerbosity",
                    help="Console message verbosity: 0 => none")
  (options, args) = parser.parse_args(sys.argv[1:])
  if len(args) < 5:
    parser.print_help(sys.stderr)
    sys.exit()

  with open(args[3]) as paramsFile:
    params = yaml.safe_load(paramsFile)
  return options, args, params



if __name__ == "__main__":
  (options, args, prms) = _getArgs()
  run(int(args[0]), int(args[1]), int(args[2]), prms, args[4],
      options.plotVerbosity, options.consoleVerbosity)
