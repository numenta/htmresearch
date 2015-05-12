import unittest

import numpy

from union_pooling.union_pooler import UnionPooler



REAL_DTYPE = numpy.float32



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


  def testDecayPoolingActivationDefaultDecayRate(self):
    self.unionPooler._poolingActivation = numpy.array([0, 1, 2, 3, 4],
                                                      dtype=REAL_DTYPE)
    expected = numpy.array([0, 0, 1, 2, 3], dtype=REAL_DTYPE)

    result = self.unionPooler._decayPoolingActivation()

    self.assertTrue(numpy.array_equal(expected, result))


  def testDecayPoolingActivationSpecifiedDecayRate(self):
    self.unionPooler._decayFunctionSlope = 10
    self.unionPooler._poolingActivation = numpy.array([0, 10, 20, 30, 40],
                                                      dtype=REAL_DTYPE)
    expected = numpy.array([0, 0, 10, 20, 30], dtype=REAL_DTYPE)

    result = self.unionPooler._decayPoolingActivation()

    self.assertTrue(numpy.array_equal(expected, result))


  def testAddToPoolingActivation(self):
    activeCells = numpy.array([1, 3, 4])
    #                      [    0,   1,   0,     1,     1]
    overlaps = numpy.array([0.123, 0.0, 0.0, 0.456, 0.789])
    expected = [0.0, 0.0, 0.0, 0.456, 0.789]

    result = self.unionPooler._addToPoolingActivation(activeCells, overlaps)

    self.assertTrue(numpy.allclose(expected, result))


  def testAddToPoolingActivationExistingActivation(self):
    self.unionPooler._poolingActivation = numpy.array([0, 1, 2, 3, 4],
                                                      dtype=REAL_DTYPE)
    activeCells = numpy.array([1, 3, 4])
    #                      [    0,   1,   0,     1,     1]
    overlaps = numpy.array([0.123, 0.0, 0.0, 0.456, 0.789])
    expected = [0.0, 1.0, 2.0, 3.456, 4.789]

    result = self.unionPooler._addToPoolingActivation(activeCells, overlaps)

    self.assertTrue(numpy.allclose(expected, result))


  def testGetMostActiveCellsUnionSizeZero(self):
    self.unionPooler._poolingActivation = numpy.array([0, 1, 2, 3, 4],
                                                      dtype=REAL_DTYPE)
    self.unionPooler._maxUnionCells = 0

    result = self.unionPooler._getMostActiveCells()

    self.assertEquals(len(result), 0)


  def testGetMostActiveCellsRegular(self):
    self.unionPooler._poolingActivation = numpy.array([0, 1, 2, 3, 4],
                                                      dtype=REAL_DTYPE)

    result = self.unionPooler._getMostActiveCells()

    self.assertEquals(len(result), 1)
    self.assertEquals(result[0], 4)


  def testGetMostActiveCellsIgnoreZeros(self):
    self.unionPooler._poolingActivation = numpy.array([0, 0, 0, 3, 4],
                                                      dtype=REAL_DTYPE)
    self.unionPooler._maxUnionCells = 3

    result = self.unionPooler._getMostActiveCells()

    self.assertEquals(len(result), 2)
    self.assertEquals(result[0], 4)
    self.assertEquals(result[1], 3)


if __name__ == "__main__":
  unittest.main()
