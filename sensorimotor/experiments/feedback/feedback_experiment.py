import time
import sys
import yaml
from optparse import OptionParser


RANDOM_SEED = 42



def setupExperiment(n, w, worldSequences,  worldSequenceLength, numElements):
  # print "Setting up experiment..."
  # universe = OneDUniverse(nSensor=n, wSensor=w,
  #                         nMotor=n, wMotor=w)
  # exhaustiveAgents = []
  # randomAgents = []
  # for world in xrange(numWorlds):
  #   elements = range(world * numElements, world * numElements + numElements)
  #   agent = ExhaustiveOneDAgent(OneDWorld(universe, elements), 0)
  #   exhaustiveAgents.append(agent)
  #
  #   possibleMotorValues = range(-numElements, numElements + 1)
  #   possibleMotorValues.remove(0)
  #   agent = RandomOneDAgent(OneDWorld(universe, elements), numElements / 2,
  #                           possibleMotorValues=possibleMotorValues,
  #                           seed=RANDOM_SEED)
  #   randomAgents.append(agent)
  # print "Done setting up experiment."
  # print
  # return exhaustiveAgents, randomAgents
  pass


def run(worldSequences, worldSequenceLength, numElements, paramsDict, outputDir,
        plotVerbosity, consoleVerbosity):
  # Setup params
  n = params["n"]
  w = params["w"]
  tmParams = params["tmParams"]
  tpParams = params["tpParams"]
  isOnline = params["isOnline"]
  onlineTrainingReps = params["onlineTrainingReps"] if isOnline else None
  completeSequenceLength = numElements ** 2
  print ("Experiment parameters: "
         "(# worldSequences = {0}, # worldSequenceLength = {1}, # elements = {"
         "2}, n = {3}, w = {4}, online = {5}, onlineReps = {6})".format(
         worldSequences, worldSequenceLength, numElements, n, w, isOnline,
         onlineTrainingReps))
  print "Temporal memory parameters: {0}".format(tmParams)
  print "Temporal pooler parameters: {0}".format(tpParams)
  print

  # Setup experiment
  start = time.time()
  # runner = SensorimotorExperimentRunner(tmOverrides=tmParams,
  #                                       tpOverrides=tpParams,
  #                                       seed=RANDOM_SEED)
  trainingAgents, testAgents = setupExperiment(n, w, worldSequences,
                                               worldSequenceLength, numElements,
                                               tmParams, tpParams)

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
  (options, args, params) = _getArgs()
  run(int(args[0]), int(args[1]), int(args[2]), params, args[4],
      options.plotVerbosity, options.consoleVerbosity)
