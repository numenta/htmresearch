import numpy
import random

from htmresearch.algorithms.column_pooler import ColumnPooler
from htmresearch.algorithms.apical_dependent_temporal_memory import ApicalDependentSequenceMemory
from feedback_sequences import generateSequences, convertSequenceMachineSequence



def getDefaultL4Params(inputSize):
  """
  Returns a good default set of parameters to use in the L4 region.
  """
  return {
      "columnCount": inputSize,
      "cellsPerColumn": 8,
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
    "synPermProximalDec": 0.0033,
    "initialProximalPermanence": 0.81,
    "minThresholdProximal": 27,
    "sampleSizeProximal": 40,
    "connectedPermanenceProximal": 0.5,
    "synPermDistalInc": 0.1,
    "synPermDistalDec": 0.05,
    "initialDistalPermanence": 0.61,
    "activationThresholdDistal": 13,
    "sampleSizeDistal": 20,
    "connectedPermanenceDistal": 0.5,
    "distalSegmentInhibitionFactor": .8,
    "inertiaFactor": .9,
    "seed": seed,
  }

def test_apical_dependent_TM_learning(sequenceLen, numSequences, sharedRange, seed, training_iters):
  TM = ApicalDependentSequenceMemory(**getDefaultL4Params(2048))
  pooler = ColumnPooler(**getDefaultL2Params(2048, seed))


  print "Generating sequences..."
  sequenceMachine, generatedSequences, numbers = generateSequences(
    sequenceLength=sequenceLen, sequenceCount=numSequences,
    sharedRange=sharedRange, n = 2048, w = 40, seed=seed)

  sequences = convertSequenceMachineSequence(generatedSequences)

  pooler_representations = []
  s = 0

  characters = {}
  char_sequences = []

  sequence_order = range(numSequences)
  for i in xrange(training_iters):
    random.shuffle(sequence_order)
    for s in sequence_order:
      sequence = sequences[s]
      pooler_representation = numpy.asarray([], dtype = "int")
      TM_representation = numpy.asarray([], dtype = "int")
      char_sequences.append([])
      total_pooler_representation = set()
      t = 0
      for timestep in sequence:
        datapoint = numpy.asarray(list(timestep), dtype = "int")
        datapoint.sort()
        TM.compute(activeColumns = datapoint,
                   apicalInput = pooler_representation,
                   learn = True)
        TM_representation = TM.activeCells
        winners = TM.winnerCells
        predicted_cells = TM.predictedCells
        #megabursting = TM.megabursting
        #if i > 0:
        #  import ipdb; ipdb.set_trace()
        pooler.compute(feedforwardInput = TM_representation,
                       feedforwardGrowthCandidates = winners,
                       lateralInputs = (pooler_representation,),
                       predictedInput = predicted_cells,
                       learn = True)
        pooler_representation = pooler.activeCells
        if i == training_iters - 1 and t > 0:
          total_pooler_representation |= set(pooler_representation)
          print len(pooler_representation)
        #print pooler_representation, len(pooler_representation), (s, t)
        t += 1

      pooler.reset()
      if i == training_iters - 1:
        pooler_representations.append(total_pooler_representation)
      s += 1

  representations = pooler_representations
  #print representations
  for i in range(len(representations)):
    for j in range(i):
      print (i, j), "overlap:", len(representations[i] & representations[j]), "Length of i:", len(representations[i])

if __name__ == '__main__':
  seed = int(numpy.random.rand()*100000)
  test_apical_dependent_TM_learning(30, 5, (5, 24), seed, 15)
