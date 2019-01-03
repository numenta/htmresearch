# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2018, Numenta, Inc.  Unless you have an agreement
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

from __future__ import print_function
import unittest

import torch
from htmresearch.frameworks.pytorch.k_winners_cnn import KWinners


class TestContext(object):
  def __init__(self):
    self.saved_tensors = None

  def save_for_backward(self,x):
    self.saved_tensors = x


class KWinnersCNNTest(unittest.TestCase):
  """

  """

  def setUp(self):
    # Create vector with batch size 1, 3 filters, and width/height 2
    self.numFilters = 3
    x = torch.ones((1,3,2,2))
    x[0, 0, 1, 0] = 1.1
    x[0, 0, 1, 1] = 1.2
    x[0, 1, 0, 1] = 1.2
    x[0, 2, 1, 0] = 1.3
    self.x = x

    self.gradient = torch.rand(x.shape)

    self.dutyCycle = torch.zeros((1, self.numFilters, 1, 1))
    self.dutyCycle[:] = 1.0 / self.numFilters


  def testOne(self):
    """
    Equal duty cycle, boost factor 0, k=4
    """
    x = self.x

    ctx = TestContext()

    result = KWinners.forward(ctx, x, self.dutyCycle, k=4, boostStrength=0.0)

    expected = torch.zeros_like(x)
    expected[0, 0, 1, 0] = 1.1
    expected[0, 0, 1, 1] = 1.2
    expected[0, 1, 0, 1] = 1.2
    expected[0, 2, 1, 0] = 1.3

    self.assertEqual(result.shape, expected.shape)

    numCorrect = (result == expected).sum()
    self.assertEqual(numCorrect, result.reshape(-1).size()[0])

    indices = ctx.saved_tensors.reshape(-1)
    expectedIndices = torch.tensor([2,  3, 10,  5])
    numCorrect = (indices == expectedIndices).sum()
    self.assertEqual(numCorrect, 4)

    # Test that gradient values are in the right places, that their sum is
    # equal, and that they have exactly the right number of nonzeros
    grad_x, _, _, _ = KWinners.backward(ctx, self.gradient)
    grad_x = grad_x.reshape(-1)
    self.assertEqual(
      (grad_x[indices] == self.gradient.reshape(-1)[indices]).sum(), 4)
    self.assertAlmostEqual(
      grad_x.sum(), self.gradient.reshape(-1)[indices].sum(), places=4)
    self.assertEqual(len(grad_x.nonzero()), 4)


  def testTwo(self):
    """
    Equal duty cycle, boost factor 0, k=3
    """
    x = self.x

    ctx = TestContext()

    result = KWinners.forward(ctx, x, self.dutyCycle, k=3, boostStrength=0.0)

    expected = torch.zeros_like(x)
    expected[0, 0, 1, 1] = 1.2
    expected[0, 1, 0, 1] = 1.2
    expected[0, 2, 1, 0] = 1.3

    self.assertEqual(result.shape, expected.shape)

    numCorrect = (result == expected).sum()
    self.assertEqual(numCorrect, result.reshape(-1).size()[0])

    indices = ctx.saved_tensors.reshape(-1)
    expectedIndices = torch.tensor([3, 10,  5])
    numCorrect = (indices == expectedIndices).sum()
    self.assertEqual(numCorrect, 3)

    # Test that gradient values are in the right places, that their sum is
    # equal, and that they have exactly the right number of nonzeros
    grad_x, _, _, _ = KWinners.backward(ctx, self.gradient)
    grad_x = grad_x.reshape(-1)
    self.assertEqual(
      (grad_x[indices] == self.gradient.reshape(-1)[indices]).sum(), 3)
    self.assertAlmostEqual(
      grad_x.sum(), self.gradient.reshape(-1)[indices].sum(), places=4)
    self.assertEqual(len(grad_x.nonzero()), 3)


  def testInitialNullInputLearnMode(self):
    """Tests with no input in the beginning. """
    x = self.x




if __name__ == "__main__":
  unittest.main()
