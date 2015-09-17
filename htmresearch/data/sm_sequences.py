#! /usr/bin/env python
# ----------------------------------------------------------------------
#  Copyright (C) 2010, Numenta Inc. All rights reserved.
#
#  The information and source code contained herein is the
#  exclusive property of Numenta Inc. No part of this software
#  may be used, reproduced, stored or distributed in any form,
#  without explicit written authorization from Numenta Inc.
# ----------------------------------------------------------------------

"""
This module helps generate encoded sensorimotor information given a grid-like
world with elements in every grid location.

Usage:
Create an instance of the SMSequences class giving the following information:
  The set of all possible sensory elements.
  A list sensory elements in the world (duplicates allowed).
  A list of coordinates corresponding to the previous list of sensory elements.
  Max and min displacement allowed in one time step for a motor transition.
  An encoding information for both sensory elements and motor commands.

Use that instance to
  Generate a sensorimotor sequence of a given length.
    SMSequences.generateSensorimotorSequence(sequenceLength)

  Encode a sensorimotor sequence given a list of coordinates.
    SMSequences.encodeSensorimotorSequence(eyeLocs)

A simple example of how you would use this class is at the bottom of this file.
Run this script with no arguments to run that code.
"""

import numpy

from nupic.bindings.math import Random

from nupic.encoders import ScalarEncoder
from nupic.encoders import VectorEncoder
from nupic.encoders.category import CategoryEncoder
from nupic.encoders.sdrcategory import SDRCategoryEncoder



# Utility routines for printing sequences
def printSequence(x, formatString="%d"):
  """
  Compact print a list or numpy array.
  """
  numElements = len(x)
  s = ""
  for j in range(numElements):
    s += formatString % x[j]
  print s

def printSequences(x, formatString="%d"):
  """
  Print a bunch of sequences stored in a 2D numpy array.
  """
  [seqLen, numElements] = x.shape
  for i in range(seqLen):
    s = ""
    for j in range(numElements):
      s += formatString % x[i][j]
    print s


class SMSequences(object):

  """
  Class generates sensorimotor sequences
  """
  def __init__(self,
               sensoryInputElements,
               spatialConfig,
               sensoryInputElementsPool=list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                                   "abcdefghijklmnopqrstuvwxyz0123456789"),
               minDisplacement=1,
               maxDisplacement=1,
               numActiveBitsSensoryInput=9,
               numActiveBitsMotorInput=9,
               seed=42,
               verbosity=False,
               useRandomEncoder=False):
    """
    @param sensoryInputElements       (list)
        Strings or numbers representing the sensory elements that exist in your
        world. Elements can be repeated if multiple of the same exist.

    @param spatialConfig              (numpy.array)
        Array of size: (1, len(sensoryInputElements), dimension). It has a
        coordinate for every element in sensoryInputElements.

    @param sensoryInputElementsPool   (list)
        List of strings representing a readable version of all possible sensory
        elements in this world. Elements don't need to be in any order and there
        should be no duplicates. By default this contains the set of
        alphanumeric characters.

    @param maxDisplacement            (int)
        Maximum `distance` for a motor command. Distance is defined by the
        largest difference along any coordinate dimension.

    @param minDisplacement            (int)
        Minimum `distance` for a motor command. Distance is defined by the
        largest difference along any coordinate dimension.

    @param numActiveBitsSensoryInput  (int)
        Number of active bits for each sensory input.

    @param numActiveBitsMotorInput    (int)
        Number of active bits for each dimension of the motor input.

    @param seed                       (int)
        Random seed for nupic.bindings.Random.

    @param verbosity                  (int)
        Verbosity

    @param useRandomEncoder           (boolean)
        if True, use the random encoder SDRCategoryEncoder. If False,
        use CategoryEncoder. CategoryEncoder encodes categories using contiguous
        non-overlapping bits for each category, which makes it easier to debug.
    """

    #---------------------------------------------------------------------------------
    # Store creation parameters
    self.sensoryInputElements = sensoryInputElements
    self.sensoryInputElementsPool = sensoryInputElementsPool
    self.spatialConfig = spatialConfig.astype(int)
    self.spatialLength = len(spatialConfig)
    self.maxDisplacement = maxDisplacement
    self.minDisplacement = minDisplacement
    self.numActiveBitsSensoryInput = numActiveBitsSensoryInput
    self.numActiveBitsMotorInput = numActiveBitsMotorInput
    self.verbosity = verbosity
    self.seed = seed

    self.initialize(useRandomEncoder)


  def initialize(self, useRandomEncoder):
    """
    Initialize the various data structures.
    """
    self.setRandomSeed(self.seed)

    self.dim = numpy.shape(self.spatialConfig)[-1]

    self.spatialMap = dict( zip( map(tuple, list(self.spatialConfig)),
                              self.sensoryInputElements))

    self.lengthMotorInput1D = (2*self.maxDisplacement + 1) * \
                                                    self.numActiveBitsMotorInput

    uniqueSensoryElements = list(set(self.sensoryInputElementsPool))

    if useRandomEncoder:
      self.sensoryEncoder = SDRCategoryEncoder(n=1024,
                                w=self.numActiveBitsSensoryInput,
                                categoryList=uniqueSensoryElements,
                                forced=True)
      self.lengthSensoryInput = self.sensoryEncoder.getWidth()

    else:
      self.lengthSensoryInput = (len(self.sensoryInputElementsPool)+1) * \
                                          self.numActiveBitsSensoryInput

      self.sensoryEncoder = CategoryEncoder(w=self.numActiveBitsSensoryInput,
                            categoryList=uniqueSensoryElements, forced=True)

    motorEncoder1D =  ScalarEncoder(n=self.lengthMotorInput1D,
                                    w=self.numActiveBitsMotorInput,
                                    minval=-self.maxDisplacement,
                                    maxval=self.maxDisplacement,
                                    clipInput=True,
                                    forced=True)

    self.motorEncoder = VectorEncoder(length=self.dim, encoder=motorEncoder1D)


  def generateSensorimotorSequence(self, sequenceLength):
    """
    Generate sensorimotor sequences of length sequenceLength.

    @param sequenceLength (int)
        Length of the sensorimotor sequence.

    @return (tuple) Contains:
            sensorySequence       (list)
                Encoded sensory input for whole sequence.

            motorSequence         (list)
                Encoded motor input for whole sequence.

            sensorimotorSequence  (list)
                Encoder sensorimotor input for whole sequence. This is useful
                when you want to give external input to temporal memory.
    """
    motorSequence = []
    sensorySequence = []
    sensorimotorSequence = []
    currentEyeLoc = self.nupicRandomChoice(self.spatialConfig)

    for i in xrange(sequenceLength):

      currentSensoryInput = self.spatialMap[tuple(currentEyeLoc)]

      nextEyeLoc, currentEyeV = self.getNextEyeLocation(currentEyeLoc)

      if self.verbosity:
          print "sensory input = ", currentSensoryInput, \
            "eye location = ", currentEyeLoc, \
            " motor command = ", currentEyeV

      sensoryInput = self.encodeSensoryInput(currentSensoryInput)
      motorInput = self.encodeMotorInput(list(currentEyeV))
      sensorimotorInput = numpy.concatenate((sensoryInput, motorInput))

      sensorySequence.append(sensoryInput)
      motorSequence.append(motorInput)
      sensorimotorSequence.append(sensorimotorInput)

      currentEyeLoc = nextEyeLoc

    return (sensorySequence, motorSequence, sensorimotorSequence)


  def encodeSensorimotorSequence(self, eyeLocs):
    """
    Encode sensorimotor sequence given the eye movements. Sequence will have
    length len(eyeLocs) - 1 because only the differences of eye locations can be
    used to encoder motor commands.

    @param eyeLocs  (list)
        Numpy coordinates describing where the eye is looking.

    @return (tuple) Contains:
            sensorySequence       (list)
                Encoded sensory input for whole sequence.

            motorSequence         (list)
                Encoded motor input for whole sequence.

            sensorimotorSequence  (list)
                Encoder sensorimotor input for whole sequence. This is useful
                when you want to give external input to temporal memory.
    """
    sequenceLength = len(eyeLocs) - 1

    motorSequence = []
    sensorySequence = []
    sensorimotorSequence = []

    for i in xrange(sequenceLength):
      currentEyeLoc = eyeLocs[i]
      nextEyeLoc = eyeLocs[i+1]

      currentSensoryInput = self.spatialMap[currentEyeLoc]

      currentEyeV = nextEyeLoc - currentEyeLoc

      if self.verbosity:
        print "sensory input = ", currentSensoryInput, \
            "eye location = ", currentEyeLoc, \
            " motor command = ", currentEyeV

      sensoryInput = self.encodeSensoryInput(currentSensoryInput)
      motorInput = self.encodeMotorInput(list(currentEyeV))
      sensorimotorInput = numpy.concatenate((sensoryInput, motorInput))

      sensorySequence.append(sensoryInput)
      motorSequence.append(motorInput)
      sensorimotorSequence.append(sensorimotorInput)

    return (sensorySequence, motorSequence, sensorimotorSequence)


  def getNextEyeLocation(self, currentEyeLoc):
    """
    Generate next eye location based on current eye location.

    @param currentEyeLoc (numpy.array)
        Current coordinate describing the eye location in the world.

    @return (tuple) Contains:
            nextEyeLoc  (numpy.array)
                Coordinate of the next eye location.

            eyeDiff     (numpy.array)
                Vector describing change from currentEyeLoc to nextEyeLoc.
    """
    possibleEyeLocs = []
    for loc in self.spatialConfig:
      shift = abs(max(loc - currentEyeLoc))
      if self.minDisplacement <= shift <= self.maxDisplacement:
        possibleEyeLocs.append(loc)

    nextEyeLoc = self.nupicRandomChoice(possibleEyeLocs)

    eyeDiff = nextEyeLoc  - currentEyeLoc

    return nextEyeLoc, eyeDiff


  def setRandomSeed(self, seed):
    """
    Reset the nupic random generator. This is necessary to reset random seed to
    generate new sequences.

    @param seed       (int)
        Seed for nupic.bindings.Random.
    """
    self.seed = seed
    self._random = Random()
    self._random.setSeed(seed)


  def nupicRandomChoice(self, array):
    """
    Chooses a random element from an array using the nupic random number
    generator.

    @param array  (list or numpy.array)
        Array to choose random element from.

    @return       (element)
        Element chosen at random.
    """
    return array[self._random.getUInt32(len(array))]


  def encodeMotorInput(self, motorInput):
    """
    Encode motor command to bit vector.

    @param motorInput (1D numpy.array)
        Motor command to be encoded.

    @return           (1D numpy.array)
        Encoded motor command.
    """
    if not hasattr(motorInput, "__iter__"):
      motorInput = list([motorInput])

    return self.motorEncoder.encode(motorInput)


  def decodeMotorInput(self, motorInputPattern):
    """
    Decode motor command from bit vector.

    @param motorInputPattern (1D numpy.array)
        Encoded motor command.

    @return                  (1D numpy.array)
        Decoded motor command.

    """
    key = self.motorEncoder.decode(motorInputPattern)[0].keys()[0]
    motorCommand = self.motorEncoder.decode(motorInputPattern)[0][key][1][0]
    return motorCommand


  def encodeSensoryInput(self, sensoryInputElement):
    """
    Encode sensory input to bit vector

    @param sensoryElement (1D numpy.array)
        Sensory element to be encoded.

    @return               (1D numpy.array)
        Encoded sensory element.
    """
    return self.sensoryEncoder.encode(sensoryInputElement)


  def decodeSensoryInput(self, sensoryInputPattern):
    """
    Decode sensory input from bit vector.

    @param sensoryInputPattern  (1D numpy.array)
        Encoded sensory element.

    @return                     (1D numpy.array)
        Decoded sensory element.
    """
    return self.sensoryEncoder.decode(sensoryInputPattern)[0]['category'][1]


  def printSensoryCodingScheme(self):
    """
    Print sensory inputs along with their encoded versions.
    """
    print "\nsensory coding scheme: "
    for loc in self.spatialConfig:
      sensoryElement = self.spatialMap[tuple(loc)]
      print sensoryElement, "%s : " % loc,
      printSequence(self.encodeSensoryInput(sensoryElement))


  def printMotorCodingScheme(self):
    """
    Print motor commands (displacement vector) along with their encoded
    versions.
    """
    print "\nmotor coding scheme: "
    self.build(self.dim, [])


  def build(self, n, vec):
    """
    Recursive function to help print motor coding scheme.
    """
    for i in range(-self.maxDisplacement, self.maxDisplacement+1):
      next = vec + [i]
      if n == 1:
        print '{:>5}\t'.format(next), " = ",
        printSequence(self.encodeMotorInput(next))
      else:
        self.build(n-1, next)


if __name__ == "__main__":
  # A simple example of how you would use this class

  seq = SMSequences(
    sensoryInputElementsPool=["A", "B", "C", "D", "E", "F", "G", "H"],
    sensoryInputElements=["E", "D", "A", "D", "G", "H"],
    spatialConfig=numpy.array([[0],[1],[2],[3],[4],[5]]),
    minDisplacement=1,
    maxDisplacement=2,
    verbosity=3,
    seed=4,
    useRandomEncoder=False
  )

  sequence = seq.generateSensorimotorSequence(10)
  print "Length of sequence:",len(sequence[0])
  for i in range(len((sequence[0]))):
    print "\n============= Sequence position",i

    # Print the sensory pattern and motor command in "English"
    print "Sensory pattern:",seq.decodeSensoryInput(sequence[0][i]),
    print "Motor command:",seq.decodeMotorInput(sequence[1][i])

    # Print the SDR's corresponding to sensory and motor commands
    print "Sensory signal",
    printSequence(sequence[0][i])
    print "Motor signal",
    printSequence(sequence[1][i])
    print "Combined distal input",
    printSequence(sequence[2][i])
