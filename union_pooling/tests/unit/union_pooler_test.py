import unittest

import numpy

from union_pooling.union_pooler import UnionPooler

class UnionPoolerTest(unittest.TestCase):


  def setUp(self):
    self.unionPooler = UnionPooler(inputDimensions=5,
                                   columnDimensions=5,
                                   potentialRadius=16,
                                   potentialPct=0.9,
                                   globalInhibition=True,
                                   localAreaDensity=-1.0,
                                   numActiveColumnsPerInhArea=2.0,
                                   stimulusThreshold=2,
                                   synPermInactiveDec=0.01,
                                   synPermActiveInc=0.03,
                                   synPermConnected=0.3,
                                   minPctOverlapDutyCycle=0.001,
                                   minPctActiveDutyCycle=0.001,
                                   dutyCyclePeriod=1000,
                                   maxBoost=1.0,
                                   seed=42,
                                   spVerbosity=0,
                                   wrapAround=True,

                                   # union_pooler.py parameters
                                   activeOverlapWeight=1.0,
                                   predictedActiveOverlapWeight=10.0,
                                   maxUnionActivity=0.20)


  def testDecayPoolingActivation(self):
    self.unionPooler._poolingActivation = numpy.array([0, 1, 2, 3, 4],
                                                      dtype="int32")
    self.unionPooler._decayPoolingActivation()

    expected = numpy.array([0, 0, 1, 2, 3])
    for i in xrange(len(expected)):
      self.assertEquals(expected[i], self.unionPooler._poolingActivation[i])


  def testAddToPoolingActivation(self):
    pass
    # activeCells = set([1, 3, 4])
    # #          [    0,   1,   0,     1,     1]
    # overlaps = [0.123, 0.0, 0.0, 0.456, 0.789]
    # expected = [0.0, 0.0, 0.0, 0.456, 0.789]
    #
    # self.unionPooler._addToPoolingActivation(activeCells, overlaps)
    #
    # for i in xrange(len(expected)):
    #   self.assertEquals(expected[i], self.unionPooler._poolingActivation[i])


  def testGetMostActiveCells(self):
    pass


if __name__ == "__main__":
  unittest.main()
