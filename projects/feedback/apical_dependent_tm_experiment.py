import numpy

from htmresearch.algorithms.column_pooler import ColumnPooler
from htmresearch.algorithms.apical_dependent_temporal_memory import ApicalDependentTemporalMemory
from feedback_sequences import generateSequences, convertSequenceMachineSequence



def getDefaultL4Params(inputSize):
  """
  Returns a good default set of parameters to use in the L4 region.
  """
  return {
      "columnCount": inputSize,
      "cellsPerColumn": 8,
      "basalInputSize": inputSize * 8,
      "apicalInputSize": 2048,
      "initialPermanence": 0.61,
      "connectedPermanence": 0.6,
      "permanenceIncrement": 0.1,
      "permanenceDecrement": 0.02,
      "minThreshold": 13,
      "basalPredictedSegmentDecrement": 0.0,
      "apicalPredictedSegmentDecrement": 0.0,
      "activationThreshold": 15,
      "sampleSize": 20,
      }

def getDefaultL2Params(inputSize, seed):
  """
  Returns a good default set of parameters to use in the L4 region.
  """
  return {
    "cellCount": 2048,
    "inputWidth": inputSize * 8,
    "lateralInputWidths": (2048,),
    "sdrSize": 40,
    "synPermProximalInc": 0.1,
    "synPermProximalDec": 0.001,
    "initialProximalPermanence": 0.81,
    "minThresholdProximal": 30,
    "sampleSizeProximal": 40,
    "connectedPermanenceProximal": 0.5,
    "synPermDistalInc": 0.1,
    "synPermDistalDec": 0.02,
    "initialDistalPermanence": 0.61,
    "activationThresholdDistal": 13,
    "sampleSizeDistal": 20,
    "connectedPermanenceDistal": 0.5,
    "distalSegmentInhibitionFactor": .8,
    "inertiaFactor": 6667.,
    "seed": seed,
  }

def test_apical_dependent_TM_learning(sequenceLen, numSequences, sharedRange, seed):
  TM = ApicalDependentTemporalMemory(**getDefaultL4Params(2048))
  pooler = ColumnPooler(**getDefaultL2Params(2048, seed))


  print "Generating sequences..."
  sequenceMachine, generatedSequences, numbers = generateSequences(
    sequenceLength=sequenceLen, sequenceCount=numSequences,
    sharedRange=sharedRange, n = 2000, w = 40, seed=seed)

  sequences = convertSequenceMachineSequence(generatedSequences)

  pooler_representations = []
  s = 0

  characters = {}
  char_sequences = []

  for sequence in sequences:
    pooler_representation = numpy.asarray([], dtype = "int")
    TM_representation = numpy.asarray([], dtype = "int")
    char_sequences.append([])
  #  for i in xrange(1):
    t = 0
    for timestep in sequence:
      #if len(pooler_representation) == 0:
      #  print t
      t += 1

      #if tuple(timestep) in characters:
      #  char_sequences[s].append(characters[tuple(timestep)])
      #else:
      #  characters[tuple(timestep)] = len(characters)
      #  char_sequences[s].append(characters[tuple(timestep)])


      datapoint = numpy.asarray(list(timestep), dtype = "int")
      datapoint.sort()
      TM.compute(activeColumns = datapoint,
                 apicalInput = pooler_representation,
                 basalInput = TM_representation,
                 learn = True)
      TM_representation = TM.activeCells
      winners = TM.winnerCells
      megabursting = TM.megabursting
      pooler.compute(feedforwardInput = TM_representation,
                     feedforwardGrowthCandidates = winners,
                     lateralInputs = (pooler_representation,),
                     bursting = megabursting,
                     learn = True)
      pooler_representation = pooler.activeCells
      if t > 0:
        pooler_representations.append(pooler_representation)
      #print pooler_representation, len(pooler_representation), (s, t)

    pooler.reset()
    s += 1

  representations = set(map(tuple, pooler_representations))
  print len(representations)
  #for representation in representations:
  #  print sorted(list(representation)), len(representation)

  print char_sequences
if __name__ == '__main__':
  test_apical_dependent_TM_learning(30, 40, (5, 24), 16)
