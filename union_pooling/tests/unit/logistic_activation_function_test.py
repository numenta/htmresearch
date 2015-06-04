import unittest

import numpy
import numpy.testing as npt

from union_pooling.activation.excite_functions.logistic_excite_function import (
  LogisticExciteFunction)


class LogisticActivationFunctionTest(unittest.TestCase):


  def setUp(self):
    self.fcn = LogisticExciteFunction(xMidpoint=5, maxValue=1, steepness=2)


  def testDecayExciteConsistency(self):
    activation = numpy.array([0.0, 0.2, 0.4, 0.6, 0.8])
    expected = numpy.copy(activation)
    amount = 1.0

    self.fcn.decay(activation, amount)
    self.fcn.excite(activation, amount)

    npt.assert_allclose(activation, expected)


  def testDecayZeroActivation(self):
    activation = numpy.array([0.0, 0.0, 0.0, 0.0, 0.0])
    expected = numpy.copy(activation)
    amount = 1.0

    self.fcn.decay(activation, amount)

    npt.assert_allclose(activation, expected)


  def testExciteFullActivation(self):
    activation = numpy.array([1.0, 1.0, 1.0, 1.0, 1.0])
    expected = numpy.copy(activation)
    amount = 10.0

    self.fcn.excite(activation, amount)

    npt.assert_allclose(activation, expected)


if __name__ == "__main__":
  unittest.main()
