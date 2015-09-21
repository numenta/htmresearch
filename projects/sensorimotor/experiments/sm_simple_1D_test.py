# An example of sensorimotor inference
# The input to CLA contains both sensory inputs ("A","B","C","D") 
# and motor commands that encodes the eye velocity vector e.g. (1,-1,2,-2,...)
# CLA will be trained on copies of valid transitions


from sensorimotor.TM_SM import TM_SM
import numpy
from sensorimotor.sm_sequences import SMSequences

"""

This program forms the simplest test of sensorimotor sequence inference
with 1D patterns. We present a sequence from a single 1D pattern. The
TM is initialized with multiple cells per column but should form a
first order representation of this sequence.

"""

# The sensory pattern has a layout as below:
# C D
# A B

sensoryInputElements = ["A", "B", "C", "D"]
spatialConfig = numpy.array([0, 1, 2, 3])

smseq = SMSequences(
  sensoryInputElements=["A", "B", "C", "D"],
  sensoryInputElementsPool=["A", "B", "C", "D"],
  spatialConfig=numpy.array([[0], [1], [2], [3]]),
  maxDisplacement=3,
  numActiveBitsSensoryInput=7,
  numActiveBitsMotorInput=7,
  verbosity=3,
  seed=1)

# print out sensory coding scheme
smseq.printSensoryCodingScheme()

# print out motor coding scheme
smseq.printMotorCodingScheme()

sequenceLength = 70
(sensorySequence, motorSequence, sensorimotorSequence) =  \
  smseq.generateSensorimotorSequence(sequenceLength)

sensoryInputWidth = smseq.lengthSensoryInput
motorCommandWidth = smseq.lengthMotorInput1D

# Create Temporal Memory instance with appropriate parameters
tm = TM_SM(numberOfCols=sensoryInputWidth,
           numberOfDistalInput=motorCommandWidth + sensoryInputWidth,
           cellsPerColumn=8,
           initialPerm=0.5,
           connectedPerm=0.6,
           minThreshold=10,
           newSynapseCount=50,
           newDistalSynapseCount=21,
           permanenceInc=0.1,
           permanenceDec=0.02,
           activationThreshold=10,
           globalDecay=0,
           burnIn=1,
           learnOnOneCell=True,
           verbosity=0)


# Step 3: send this simple sequence to the temporal memory for learning
print " Training Layer 4 temporal memory ..."

# Send each letter in the sequence in order
for j in range(sequenceLength):

  print "\nPosition:",j,"========================="
  print "Sensory pattern:",smseq.decodeSensoryInput(sensorySequence[j]),
  print "Motor command:",smseq.decodeMotorInput(motorSequence[j])
  print "Raw sensorimotor input:", sensorimotorSequence[j]
  tm.compute(sensorySequence[j], sensorimotorSequence[j], enableLearn=True,
             computeInfOutput=False)

  print "\nAll the active, predicted, and learn cells cells:"
  tm.printStates(printPrevious=False, printLearnState=True)

# Reset for learn on one cell mode
tm.reset()


# generate new sensorimotor sequence for testing
sequenceLength = 20
smseq.seed = 2
(sensorySequence, motorSequence, sensorimotorSequence) =  \
  smseq.generateSensorimotorSequence(sequenceLength)

nCorrect = 0

print " Testing temporal memory on new sensorimotor sequences with the" \
      " same static pattern..."

for j in range(sequenceLength - 1):

  tm.compute(sensorySequence[j], sensorimotorSequence[j], enableLearn=False,
             computeInfOutput=False)

  print "\nAll the active and predicted cells:"
  tm.printStates(printPrevious=False, printLearnState=False)

  # tm.getPredictedState() gets the predicted cells.
  # predictedCells[c][i] represents the state of the i'th cell in the c'th
  # column. To see if a column is predicted, we can simply take the OR
  # across all the cells in that column. In numpy we can do this by taking
  # the max along axis 1.

  predictedCells = tm.getPredictedState()
  predictedColumns = predictedCells.max(axis=1)

  # Compare the predicted columns with the current sensory inputs
  # We compute the number of matching bits for each sensory element
  matchingBits = numpy.zeros(len(smseq.sensoryInputElements), dtype="int")
  for k in range(len(smseq.sensoryInputElements)):
    matchingBits[k] = numpy.logical_and(
      smseq.encodeSensoryInput(smseq.sensoryInputElements[k]),
      predictedColumns[0:sensoryInputWidth]).sum()

  # predicted next sensory input in English
  predSensoryElement = smseq.sensoryInputElements[matchingBits.argmax()]
  # observed next sensory input in English
  nextSensoryElement = smseq.decodeSensoryInput(sensorySequence[j+1])
  # Motor command in English
  motorCommand = smseq.decodeMotorInput(motorSequence[j+1])

  if (nextSensoryElement == predSensoryElement and
      matchingBits[matchingBits.argmax()] == 7):
    nCorrect = nCorrect + 1


  print " Next sensory Input: ", nextSensoryElement, " Predicted: ", \
    predSensoryElement, "with overlap=",matchingBits[matchingBits.argmax()],\
    "Motor Command: ", motorCommand

print "Correct Rate is ", float(nCorrect) / float(sequenceLength)
