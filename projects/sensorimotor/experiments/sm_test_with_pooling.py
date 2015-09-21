# An example of sensorimotor inference
# The input to CLA contains both sensory inputs ("A","B","C","D") 
# and motor commands that encodes the eye velocity vector e.g. (1,-1,2,-2,...)
# CLA will be trained on copies of valid transitions

import sys
from sensorimotor.TM_SM import TM_SM
from sensorimotor.temporal_pooler import TemporalPooler

import numpy 

from sensorimotor.sm_sequences import SMSequences, printSequence, printSequences
from nupic.bindings.math import (SM32 as SparseMatrix,
                                SM_01_32_32 as SparseBinaryMatrix,
                                GetNTAReal,
                                Random as NupicRandom)

from nupic.encoders.category import CategoryEncoder

realDType = GetNTAReal()
uintType = "uint32"

VERBOSITY = 0

"""

This program simulates temporal pooling ideas. The goal of this script is to
test sensorimotor inference and pooling in the case where
we have shared patterns.

The sensory pattern has a layout as below:
A B C D
or
D C B A

Two distinct sensorimotor sequences are generated from the two patterns

Layer 4 learns both sensorimotor patterns, however, it cannot
accurately predict the next sensory input. If the current input is "B"
and motor command is "move to right by 1", both C and A will be predicted

The layer 3 performs temporal pooling, same set of cells will be activated
as long as same sensorimotor pattern is shown, yet different set of cells
will be activated for the two patterns

"""

def formatRow(x, formatString = "%d", rowSize = 700):
  """
  Utility routine for pretty printing large vectors
  """
  s = ''
  for c,v in enumerate(x):
    if c > 0 and c % 7 == 0:
      s += ' '
    if c > 0 and c % rowSize == 0:
      s += '\n'
    s += formatString % v
  s += ' '
  return s

def extractInputForTP(tm, l3InputSize):
  # correctly predicted cells in layer 4
  correctlyPredictedCells = numpy.zeros(l3InputSize).astype(realDType)
  idx = (tm.predictedState['t-1'] + tm.activeState['t']) == 2
  idx = idx.reshape(l3InputSize)
  correctlyPredictedCells[idx] = 1.0
  # print "Predicted->active cell indices=",correctlyPredictedCells.nonzero()[0]

  # all currently active cells in layer 4
  spInputVector = tm.activeState['t'].reshape(l3InputSize)

  # bursting cells in layer 4
  burstingColumns = tm.activeState['t'].sum(axis=1)
  burstingColumns[ burstingColumns < tm.cellsPerColumn ] = 0
  burstingColumns[ burstingColumns == tm.cellsPerColumn ] = 1

  return (correctlyPredictedCells, spInputVector, burstingColumns)



#######################################################################
#
# Step 1: initialize several static patterns. Each pattern will be represented
# by letters chosen from the alphabet.

numSensoryInputActiveBits = 25
numMotorCommandActiveBits = 25
useRandomEncoding = True
maxDisplacement = 3

# spatial configuration of the pattern
spatialConfig = numpy.array([0,1,2,3],dtype='int32')

print " Generating sequences for each pattern:  "

smseqs = [
  SMSequences(
    sensoryInputElements = ['A', 'B', 'C', 'D'],
    spatialConfig = numpy.array([[0],[1],[2],[3]]),
    numActiveBitsSensoryInput = numSensoryInputActiveBits,
    numActiveBitsMotorInput = numMotorCommandActiveBits,
    maxDisplacement=maxDisplacement,
    verbosity = 3,
    seed = 1,
    useRandomEncoder=useRandomEncoding),
  SMSequences(
    sensoryInputElements = ['A', 'B', 'C', 'D'],
    spatialConfig = numpy.array([[3],[2],[1],[0]]),
    numActiveBitsSensoryInput = numSensoryInputActiveBits,
    numActiveBitsMotorInput = numMotorCommandActiveBits,
    maxDisplacement=maxDisplacement,
    verbosity = 3,
    seed = 1,
    useRandomEncoder=useRandomEncoding),
  SMSequences(
    sensoryInputElements = ['E', 'F', 'G', 'H'],
    spatialConfig = numpy.array([[0],[1],[2],[3]]),
    numActiveBitsSensoryInput = numSensoryInputActiveBits,
    numActiveBitsMotorInput = numMotorCommandActiveBits,
    maxDisplacement=maxDisplacement,
    verbosity = 3,
    seed = 1,
    useRandomEncoder=useRandomEncoding),
  SMSequences(
    sensoryInputElements = ['E', 'F', 'G', 'H'],
    spatialConfig = numpy.array([[3],[2],[1],[0]]),
    numActiveBitsSensoryInput = numSensoryInputActiveBits,
    numActiveBitsMotorInput = numMotorCommandActiveBits,
    maxDisplacement=maxDisplacement,
    verbosity = 3,
    seed = 1,
    useRandomEncoder=useRandomEncoding),
  SMSequences(
    sensoryInputElements = ['I', 'J', 'K', 'L'],
    spatialConfig = numpy.array([[3],[2],[1],[0]]),
    numActiveBitsSensoryInput = numSensoryInputActiveBits,
    numActiveBitsMotorInput = numMotorCommandActiveBits,
    maxDisplacement=maxDisplacement,
    verbosity = 3,
    seed = 1,
    useRandomEncoder=useRandomEncoding),
  SMSequences(
    sensoryInputElements = ['I', 'J', 'K', 'L'],
    spatialConfig = numpy.array([[0],[1],[2],[3]]),
    numActiveBitsSensoryInput = numSensoryInputActiveBits,
    numActiveBitsMotorInput = numMotorCommandActiveBits,
    maxDisplacement=maxDisplacement,
    verbosity = 3,
    seed = 1,
    useRandomEncoder=useRandomEncoding),
  SMSequences(
    sensoryInputElements = ['A', 'E', 'E', 'J'],
    spatialConfig = numpy.array([[0],[1],[2],[3]]),
    numActiveBitsSensoryInput = numSensoryInputActiveBits,
    numActiveBitsMotorInput = numMotorCommandActiveBits,
    maxDisplacement=maxDisplacement,
    verbosity = 3,
    seed = 1,
    useRandomEncoder=useRandomEncoding)
]

numberOfPatterns = len(smseqs)

for i,seq in enumerate(smseqs):
  print "Pattern: ",i,":", seq.sensoryInputElements,seq.spatialConfig


# Step 2: generate sensorimotor sequences

sequenceLength = 60
sensorySequences = []
motorSequences = []
sensorimotorSequences = []

for i,smseq in enumerate(smseqs):
  print "\n\nSequence",i
  (sensorySequence, motorSequence, sensorimotorSequence) = \
              smseq.generateSensorimotorSequence(sequenceLength)
  sensorySequences.append(sensorySequence)
  motorSequences.append(motorSequence)
  sensorimotorSequences.append(sensorimotorSequence)

sensoryInputWidth = smseqs[0].lengthSensoryInput
motorCommandWidth = smseqs[0].lengthMotorInput1D

print "motorCommandWidth",motorCommandWidth

# Create Temporal memory instance with appropriate parameters
# It performs sensorimotor inference, simulating layer 4
tm = TM_SM(numberOfCols=sensoryInputWidth,
        numberOfDistalInput=motorCommandWidth+sensoryInputWidth,
        cellsPerColumn=8,
        initialPerm=0.4,
        connectedPerm=0.5,
        minThreshold=45,
        activationThreshold=45,
        newSynapseCount=50,
        newDistalSynapseCount=50,
        permanenceInc=0.1,
        permanenceDec=0.02,
        learnOnOneCell=True,
        learnDistalInputs=True,
        learnLateralConnections=False,
        globalDecay=0,
        burnIn=1,
        verbosity = 0)

print "\nLayer 4 TM parameters:"
tm.printParameters()


#######################################################################
#
# Step 3: Send SM sequences to the temporal memory for learning

print " Training temporal memory (layer 4) for sensorimotor sequence  ..."

for i,smseq in enumerate(smseqs):

  sensorySequence = sensorySequences[i]
  motorSequence = motorSequences[i]
  sensorimotorSequence = sensorimotorSequences[i]

  print " Presenting Sequence ", i

  # The reset command is necessary when learnOnOneCell is True.
  if tm.learnOnOneCell:
    tm.reset()

  for j in range(sequenceLength):

    print "\nSequence:",i,"Position:",j,"========================="
    print "Sensory pattern:",smseq.decodeSensoryInput(sensorySequence[j]),
    print "Motor command:",smseq.decodeMotorInput(motorSequence[j])
    if VERBOSITY >= 2:
      print "Raw sensory input:",sensorySequence[j]
      print "Raw motor input:",motorSequence[j]
      print "Raw sensorimotor input:", sensorimotorSequence[j]

    tm.compute(sensorySequence[j], sensorimotorSequence[j], enableLearn = True,
               computeInfOutput = False)

    #tm.printStates(printPrevious=False, printLearnState=True)

    # sensoryInput = smseq.decodeSensoryInput(sensorySequence[j,:])
    # motorCommand = smseq.decodeMotorInput(motorSequence[j,:])
    # print " sensory input: ", sensoryInput, " motor command, ", motorCommand

  # tm.printCells()

# after learning, each dendritic segment represent a sensorimotor transition
# each cell represent one or more sensorimotor transitions.
# each sensorimotor transition will activate no more than one cell per column

# the following code analyze connectivity in layer 4

# for c in xrange(tm.numberOfCols):
#   print
#   for i in xrange(tm.cellsPerColumn):
#     for j,s in enumerate(tm.cells[c][i]):
#       sortedSyns = sorted(s.dsyns)
#       smConnection = numpy.zeros((sensoryInputWidth + motorCommandWidth),dtype = 'int32')
#       for _, synapse in enumerate(sortedSyns):
#         if synapse[2]>0:
#           smConnection[synapse[0]] = 1
#       SensoryConnection = smseq.decodeSensoryInput(smConnection[range(sensoryInputWidth)])
#       MotorConnection = smseq.decodeMotorInput(smConnection[range(sensoryInputWidth,sensoryInputWidth+motorCommandWidth)])
#       print 'col: ',c, ' representing ', sensoryInputElements[c/smseq.numActiveBitsSensoryInput - 1],\
#           ' cell: ', i, ' segment: ',j, ' sensory: ', SensoryConnection, ' motor: ', MotorConnection


print "\n-------------- DONE TRAINING SENSORIMOTOR SEQUENCE LEARNER -----------------"


# Layer 3 Temporal Pooler.

# Inputs to the layer 3 temporal pooler are the cells from the temporal pooler
# Temporal pooler cell [column c, cell i] corresponds to input
# c * cellsPerCol + i

print "Initializing Temporal Pooler"
l3NumColumns = 512
l3NumActiveColumnsPerInhArea = 20
l3InputSize = tm.numberOfCols*tm.cellsPerColumn
l3sp = TemporalPooler(
      inputDimensions  = [l3InputSize],
      columnDimensions = [l3NumColumns],
      potentialRadius  = l3InputSize,
      globalInhibition = True,
      numActiveColumnsPerInhArea=l3NumActiveColumnsPerInhArea,
      synPermInactiveDec=0,
      synPermActiveInc=0.001,
      synPredictedInc = 0.5,
      maxBoost=1.0,
      seed=4,
      potentialPct=0.9,
      stimulusThreshold = 2,
      useBurstingRule = False,
      minPctActiveDutyCycle = 0.1,
      synPermConnected = 0.3,
      initConnectedPct=0.2,
      spVerbosity=0
    )

print "Layer 3 Temporal Pooler parameters:"
l3sp.printParameters()

#######################################################################
#
# Step 4: Train temporal pooler (layer 3) to form stable and distinct
# representation for sensorimotor sequence

print "\n\n\n===================================================\n"
print "Running learning with the TP (layer 3) "

l3ActivityLearning = numpy.zeros((numberOfPatterns, sequenceLength,
                                  l3sp._numColumns ),dtype = "int32")

for s in range(numberOfPatterns):
  if VERBOSITY > 1:
    print
    print "Training pooling with spatial pattern",s
    print "\n-------------- START SEQUENCE -----------------"

  sensorySequence = sensorySequences[s]
  motorSequence = motorSequences[s]
  sensorimotorSequence = sensorimotorSequences[s]

  tm.reset()

  for j in range(sequenceLength):

    if VERBOSITY >= 2:
      print "\n\n-------- [Spatial Pattern,Sequence Position]:",[s,j]
      print "Sensory pattern:",smseq.decodeSensoryInput(sensorySequence[j]),
      print "Motor command:",smseq.decodeMotorInput(motorSequence[j])
      print "Raw sensory input vector:\n", \
            formatRow(sensorySequence[j].nonzero()[0], formatString="%4d")
      print "Raw motor input vector\n",formatRow(motorSequence[j].nonzero()[0],
                                                 formatString="%4d")
      print "Raw sensorimotor vector:\n", \
            formatRow(sensorimotorSequence[j].nonzero()[0], formatString="%4d")

    # Send each vector to the TM (layer 4), with learning turned off.
    # print "-------- Sequence learner (layer 4) --------:"
    tm.compute(sensorySequence[j], sensorimotorSequence[j],
               enableLearn = False, computeInfOutput = False)

    (correctlyPredictedCells, spInputVector, burstingColumns) = \
        extractInputForTP(tm, l3InputSize)

    # print "-------- Temporal Pooling (Layer 3) ---------:"
    # activeArray will hold the TemporalPooler output
    activeArray = numpy.zeros(l3sp._numColumns)

    l3sp.compute(spInputVector, learn=True, activeArray=activeArray,
                 burstingColumns=burstingColumns,
                 predictedCells=correctlyPredictedCells)
    l3ActivityLearning[s,j,:] = activeArray
    if VERBOSITY >= 2:
      print "L4 Active Cells \n",formatRow(spInputVector.nonzero()[0],
                                           formatString="%4d")
      print "L4 correctly predicted Cells \n",\
        formatRow(correctlyPredictedCells.nonzero()[0],
        formatString="%4d")
      print "L3 Active Cells \n",formatRow(activeArray.nonzero()[0],
                                           formatString="%4d")


for s in range(numberOfPatterns):
  print " L3 Activity for Sensorimotor Sequence ", s
  for j in range(l3ActivityLearning.shape[1]-10,l3ActivityLearning.shape[1]):
    print " element, ",j,
    print formatRow( l3ActivityLearning[s][j] )

# Use the stable activity for each pattern as the "template"
l3ActivityForPattern = numpy.zeros((numberOfPatterns, l3sp._numColumns ),
                                   dtype = "int32")
for s in range(numberOfPatterns):
  l3ActivityForPattern[s] = l3ActivityLearning[s][l3ActivityLearning.shape[1]-1]
  print "L3 Representation for Pattern ",s," is ",\
    formatRow(l3ActivityForPattern[s].nonzero()[0], formatString="%4d")


#######################################################################
#
# Step 5: Test temporal pooler on new random sequences with the same
# spatial configuration

print "\n\n\n===================================================\n"
print "Turn off learning of temporal pooling, and test on new sequences "
numSharedBits = numpy.zeros((numberOfPatterns,numberOfPatterns))
nRepeat = 1;
sequenceLength = 10
for repeat in range(nRepeat):

  #  Generate new SM sequence with different random seed
  sensorySequences = []
  motorSequences = []
  sensorimotorSequences = []
  for i in range(numberOfPatterns):
    smseqs[i].setRandomSeed(repeat*numberOfPatterns+i)

    (sensorySequence, motorSequence, sensorimotorSequence) = smseqs[
      i].generateSensorimotorSequence(sequenceLength)
    sensorySequences.append(sensorySequence)
    motorSequences.append(motorSequence)
    sensorimotorSequences.append(sensorimotorSequence)


  l3Activity = numpy.zeros((numberOfPatterns, sequenceLength,
                            l3sp._numColumns), dtype = "int32")

  for s in range(numberOfPatterns):
    if VERBOSITY > 1:
      print
      print
      print "Testing pooling with sequence",s
      print "\n-------------- START SEQUENCE -----------------"

    sensorySequence = sensorySequences[s]
    motorSequence = motorSequences[s]
    sensorimotorSequence = sensorimotorSequences[s]

    # no need to reset at inference stage
    tm.reset()
    l3sp._previousActiveColumns = []

    for j in range(sequenceLength):

      if VERBOSITY >= 2:
        print "\n\n-------- [Spatial Pattern,Sequence Position]:",[s,j]
        print "Sensory pattern:",smseq.decodeSensoryInput(sensorySequence[j]),
        print "Motor command:",smseq.decodeMotorInput(motorSequence[j])
        print "Raw sensory input vector:\n", \
              formatRow(sensorySequence[j].nonzero()[0], formatString="%4d")
        print "Raw motor input vector\n",formatRow(motorSequence[j].nonzero()[0],
                                                   formatString="%4d")
        print "Raw sensorimotor vector:\n", \
              formatRow(sensorimotorSequence[j].nonzero()[0], formatString="%4d")

        print "-------- Sequence learner (layer 4) --------:"

      # Send each vector to the TM (layer 4), with learning turned off
      tm.compute(sensorySequence[j], sensorimotorSequence[j],
                 enableLearn = False, computeInfOutput = False)

      (correctlyPredictedCells, spInputVector, burstingColumns) = \
        extractInputForTP(tm, l3InputSize)

      # print "-------- Temporal Pooling (Layer 3) ---------:"
      activeArray = numpy.zeros(l3sp._numColumns)      
      
      # activeArray will hold the TemporalPooler output
      l3sp.compute(spInputVector, learn=False, activeArray=activeArray,
                   burstingColumns=burstingColumns,
                   predictedCells=correctlyPredictedCells)
      l3Activity[s,j,:] = activeArray

      if VERBOSITY >= 2:
        print "L4 Active Cells \n",formatRow(spInputVector.nonzero()[0],
                                             formatString="%4d")
        print "L4 correctly predicted Cells \n",\
          formatRow(correctlyPredictedCells.nonzero()[0],
          formatString="%4d")
        print "L3 Active Cells \n",formatRow(activeArray.nonzero()[0],
                                             formatString="%4d")

  # Print out the L3 representations for each pattern for each point in the
  # sequence.
  if VERBOSITY >= 1:
    for s in range(numberOfPatterns):
      print "\nLast stable L3 activity for pattern",s,formatRow(
        l3ActivityForPattern[s])
      print " L3 Activity for Sensorimotor Sequence ", s
      for j in range(sequenceLength):
        print " element, ",j,
        print formatRow( l3Activity[s][j] )

  # For this repetition, Compute the total number of shared bits between all
  # pairs of patterns, for each point in the sequence (except we skip the
  # first element in the sequence)
  for p1 in range(numberOfPatterns):
    for p2 in range(numberOfPatterns):
      for j1 in range(1,sequenceLength):
        for j2 in range(1,sequenceLength):
          sharedBits = numpy.logical_and(l3Activity[p1][j1],
                                         l3Activity[p2][j2]).sum()
          numSharedBits[p1][p2] += sharedBits
          if VERBOSITY >=2:
            print p1,p2,j1,j2,sharedBits
            if p1 != p2 and sharedBits > 3:
              print formatRow(l3Activity[p1][j1])
              print formatRow(l3Activity[p2][j2])



# Print summary statistics
numComparisons = nRepeat*(sequenceLength-1)*(sequenceLength-1)
AvgNumSharedBits = numSharedBits/numComparisons
print " After the first few elements in the SM sequence"
print " L3 activity is stable during presentation of the same sensorimotor sequence "
print " Different L3 cells are activated for different sensorimotor sequences "

print "numComparisons=",numComparisons
print "Total number of shared bits=\n",numSharedBits
print "Average number of shared bits=\n"
for row in range(numberOfPatterns):
  print formatRow(AvgNumSharedBits[row], "%6.2f  ")

AvgSharedBitsCorrect = numpy.trace(AvgNumSharedBits)/numberOfPatterns
AvgSharedBitsIncorrect = (AvgNumSharedBits.sum() - numpy.trace(AvgNumSharedBits)) \
                          /(numberOfPatterns**2 - numberOfPatterns)

print " Shared Bits Incorrect / Correct = ", AvgSharedBitsIncorrect/AvgSharedBitsCorrect
