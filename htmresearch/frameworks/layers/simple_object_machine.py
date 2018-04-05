# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2016, Numenta, Inc.  Unless you have an agreement
# with Numenta, Inc., for a separate license for this software code, the
# following terms and conditions apply:
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero Public License version 3 as
# published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU Affero Public License for more details.
#
# You should have received a copy of the GNU Affero Public License
# along with this program.  If not, see http://www.gnu.org/licenses.
#
# http://numenta.org/licenses/
# ----------------------------------------------------------------------

import random
import numpy

from htmresearch.frameworks.layers.object_machine_base import ObjectMachineBase



class SimpleObjectMachine(ObjectMachineBase):
  """
  Most basic implementation of ObjectMachine, where objects are defined by set
  of pairs of indices, corresponding to (location, feature) pairs.
  Each feature (resp. location) index corresponds to a randomly-generated SDR
  in the machine.
  """


  def __init__(self,
               numInputBits=40,
               sensorInputSize=2048,
               externalInputSize=2048,
               numCorticalColumns=1,
               numLocations=10000,
               numFeatures=400,
               seed=42):
    """
    At creation, the SimpleObjectMachine creates a pool of locations and
    features SDR's.

    Parameters:
    ----------------------------
    @param   numInputBits (int)
             Number of ON bits in the input

    @param   sensorInputSize (int)
             Total number of bits in the sensory input

    @param   externalInputSize (int)
             Total number of bits the external (location) input

    @param   numCorticalColumns (int)
             Number of cortical columns used in the experiment

    @param   numLocations (int)
             Number of location SDRs to generate per cortical column. There is
             typically no need to not use the default value, unless the user
             knows he will use more than 10000 patterns.

    @param   numFeatures (int)
             Number of feature SDRs to generate per cortical column. There is
             typically no need to not use the default value, unless the user
             knows he will use more than 400 patterns.

    @param   seed (int)
             Seed to be used in the machine

    """
    super(SimpleObjectMachine, self).__init__(numInputBits,
                                              sensorInputSize,
                                              externalInputSize,
                                              numCorticalColumns,
                                              seed)

    # location and features pool
    self.seed = seed
    self.numLocations = numLocations
    self.numFeatures = numFeatures
    self._generateLocations()
    self._generateFeatures()
    numpy.random.seed(seed)


  def provideObjectsToLearn(self, objectNames=None):
    """
    Returns the objects in a canonical format to be sent to an experiment.

    The returned format is a a dictionary where the keys are object names, and
    values are lists of sensations, each sensation being a mapping from
    cortical column index to a pair of SDR's (one location and one feature).

    returnDict = {
      "objectId1": [
        {
          0: (set([1, 5, 10]), set([6, 12, 52]),  # location, feature for CC0
          1: (set([6, 2, 15]), set([64, 1, 5]),  # location, feature for CC1
        },
        {
          0: (set([5, 46, 50]), set([8, 10, 11]),  # location, feature for CC0
          1: (set([1, 6, 45]), set([12, 17, 23]),  # location, feature for CC1
        },
      ],
      "objectId2": [
        :
      ]
      :
    }


    Parameters:
    ----------------------------
    @param   objectNames (list)
             List of object names to provide to the experiment

    """
    if objectNames is None:
      objectNames = self.objects.keys()

    objects = {}
    for name in objectNames:
      objects[name] = [self._getSDRPairs([pair] * self.numColumns) \
                       for pair in self.objects[name]]

    self._checkObjectsToLearn(objects)
    return objects


  def provideObjectToInfer(self, inferenceConfig):
    """
    Returns the sensations in a canonical format to be sent to an experiment.

    The input inferenceConfig should be a dict with the following form:
    {
      "numSteps": 2  # number of sensations for each column
      "pairs": {
        0: [(1, 2), (2, 2)]  # sensations for cortical column 0
        1: [(2, 2), (1, 1)]  # sensations for cortical column 1
      }

      # The following are optional
      "noiseLevel": 0.05            # noise to add to feature sensations
      "locationNoise": 0.23         # noise to add to location signal
      "includeRandomLocation": True # Swap in a random location SDR
      "numAmbiguousLocations": 2    # Number of additional random locations to
                                    # union together in location input
    }

    The pairs of indices can be modified for custom inference:
      - a tuple instead of an index indicates that the union of designated
      patterns are being sensed (either in location or feature)
      - -1 as an index indicates that the input is empty for a feature, and
      is random for a location (since an empty location makes the layer 4
      burst for now)

    The returned format is a a lists of sensations, each sensation being a
    mapping from cortical column index to a pair of SDR's (one location and
    one feature).

    Parameters:
    ----------------------------
    @param   inferenceConfig (dict)
             Inference spec for experiment (cf above for format)

    """
    numSteps = inferenceConfig.get("numSteps",
                                   len(inferenceConfig["pairs"][0]))

    # some checks
    if numSteps == 0:
      raise ValueError("No inference steps were provided")
    for col in xrange(self.numColumns):
      if len(inferenceConfig["pairs"][col]) != numSteps:
        raise ValueError("Incompatible numSteps and actual inference steps")

    sensationSteps = []
    for step in xrange(numSteps):
      pairs = [
        inferenceConfig["pairs"][col][step] for col in xrange(self.numColumns)
      ]
      sdrPairs = self._getSDRPairs(
        pairs,
        noise=inferenceConfig.get("noiseLevel", None),
        locationNoise=inferenceConfig.get("locationNoise", None),
        includeRandomLocation=inferenceConfig.get("includeRandomLocation",
                                                  False),
        numAmbiguousLocations=inferenceConfig.get("numAmbiguousLocations", 0))
      sensationSteps.append(sdrPairs)

    self._checkObjectToInfer(sensationSteps)
    return sensationSteps


  def addObject(self, pairs, name=None):
    """
    Adds an object to the Machine.
    """
    if name is None:
      name = len(self.objects)

    self.objects[name] = pairs


  def createRandomObjects(self,
                          numObjects,
                          numPoints,
                          numLocations=None,
                          numFeatures=None):
    """
    Creates a set of random objects and adds them to the machine.
    If numLocations and numFeatures and not specified, they will be set to the
    desired number of points.
    """
    if numObjects > 0:
      if numLocations is None:
        numLocations = numPoints
      if numFeatures is None:
        numFeatures = numPoints

      assert(numPoints <= numLocations), ("Number of points in object cannot be "
            "greater than number of locations")

      locationArray = numpy.array(range(numLocations))
      numpy.random.seed(self.seed)
      for _ in xrange(numObjects):
        # Permute the number of locations and select points from it
        locationArray = numpy.random.permutation(locationArray)
        self.addObject(
          [(locationArray[p],
            numpy.random.randint(0, numFeatures)) for p in xrange(numPoints)],
        )


  def getDistinctPairs(self):
    """
    Return a set consisting of unique feature/location pairs across all
    objects
    """
    distinctPairs = set()
    for pairs in self.objects.itervalues():
      distinctPairs = distinctPairs.union(set(pairs))
    return distinctPairs


  def getUniqueFeaturesLocationsInObject(self, name):
    """
    Return two sets. The first set contains the unique locations Ids in the
    object. The second set contains the unique feature Ids in the object.
    """
    uniqueFeatures = set()
    uniqueLocations = set()
    for pair in self.objects[name]:
      uniqueLocations = uniqueLocations.union({pair[0]})
      uniqueFeatures = uniqueFeatures.union({pair[1]})

    return uniqueLocations, uniqueFeatures


  def _getSDRPairs(self,
                   pairs,
                   noise=None,
                   locationNoise=None,
                   includeRandomLocation=False,
                   numAmbiguousLocations=0):
    """
    This method takes a list of (location, feature) index pairs (one pair per
    cortical column), and returns a sensation dict in the correct format,
    adding noise if necessary.
    """
    sensations = {}
    numpy.random.seed(self.seed)
    for col in xrange(self.numColumns):
      locationID, featureID = pairs[col]

      # generate random location if requested
      if includeRandomLocation:
        location = self._generatePattern(self.numInputBits,
                                         self.externalInputSize)

      elif numAmbiguousLocations > 0:
        location = self.locations[col][locationID]
        for _ in range(numAmbiguousLocations):
          idx = numpy.random.randint(len(self.locations[col]))
          location = location | self.locations[col][idx]

      # generate union of locations if requested
      elif isinstance(locationID, tuple):
        location = set()
        for idx in list(locationID):
          location = location | self.locations[col][idx]
      else:
        location = self.locations[col][locationID]

      # generate empty feature if requested
      if featureID == -1:
        feature = set()
      # generate union of features if requested
      elif isinstance(featureID, tuple):
        feature = set()
        for idx in list(featureID):
          feature = feature | self.features[col][idx]
      else:
        feature = self.features[col][featureID]

      # Add random noise if requested
      if noise is not None:
        feature = self._addNoise(
          feature, noise, self.sensorInputSize)
      if locationNoise is not None:
        location = self._addNoise(
          location, locationNoise, self.externalInputSize)

      sensations[col] = (location, feature)

    return sensations


  def _addNoise(self, pattern, noiseLevel, inputSize):
    """
    Adds noise to the given pattern and returns the new one.

    A noiseLevel of 0.1 means that 10% of the ON bits will be replaced by
    other randomly chosen ON bits.  The returned SDR will still contain the
    same number of bits.

    """
    if pattern is None:
      return None

    # Bits that could be noise. These can't be from the original set.
    candidateBits = list(set(range(inputSize)) - set(pattern))
    random.shuffle(candidateBits)

    newBits = set()
    for bit in pattern:
      if random.random() < noiseLevel:
        newBits.add(candidateBits.pop())
      else:
        newBits.add(bit)

    return newBits


  def _generateLocations(self):
    """
    Generates a pool of locations to be used for the experiments.

    For each index, numColumns SDR's are created, as locations for the same
    feature should be different for each column.
    """
    size = self.externalInputSize
    bits = self.numInputBits
    random.seed(self.seed)
    self.locations = []
    for _ in xrange(self.numColumns):
      self.locations.append(
        [self._generatePattern(bits, size) for _ in xrange(self.numLocations)]
      )


  def _generateFeatures(self):
    """
    Generates a pool of features to be used for the experiments.

    For each index, numColumns SDR's are created, as locations for the same
    feature should be different for each column.
    """
    size = self.sensorInputSize
    bits = self.numInputBits
    random.seed(self.seed)
    self.features = []
    for _ in xrange(self.numColumns):
      self.features.append(
        [self._generatePattern(bits, size) for _ in xrange(self.numFeatures)]
    )
