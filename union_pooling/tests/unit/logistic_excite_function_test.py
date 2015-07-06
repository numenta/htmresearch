import unittest

import numpy
import numpy.testing as npt

from union_pooling.activation.excite_functions.excite_functions_all import (
  LogisticExciteFunction)

class LogisticExciteFunctionTest(unittest.TestCase):


  def setUp(self):
    self.fcn = LogisticExciteFunction(xMidpoint=5, minValue=10, maxValue=20, steepness=1)


  def testExcite(self):

    inputs = numpy.array([0, 2, 4, 6, 8, 10])
    original = numpy.zeros(inputs.shape)
    afterExcitation = numpy.copy(original)
    for i in xrange(len(original)):
      afterExcitation[i] = self.fcn.excite(original[i], inputs[i])

    for i in xrange(len(original)-1):
      self.assertGreater(afterExcitation[i+1], original[i])


  def testExciteZeroInputs(self):
    """
    Test saturation with strong inputs
    """
    activation = numpy.array([0, 2, 4, 6, 8])
    original = numpy.copy(activation)
    inputs = 0

    self.fcn.excite(activation, inputs)

    for i in xrange(len(original)):
      self.assertAlmostEqual(activation[i], original[i]+self.fcn._minValue)


  def testExciteFullActivation(self):
    """
    Test saturation with strong inputs
    """
    activation = numpy.array([0, 2, 4, 6, 8])
    expected = numpy.copy(activation) + self.fcn._maxValue
    inputs = 1000000
    self.fcn.excite(activation, inputs)
    npt.assert_allclose(activation, expected)


if __name__ == "__main__":
  unittest.main()
