# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2015, Numenta, Inc.  Unless you have an agreement
# with Numenta, Inc., for a separate license for this software code, the
# following terms and conditions apply:
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3 as
# published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see http://www.gnu.org/licenses.
#
# http://numenta.org/licenses/
# ----------------------------------------------------------------------

import numpy
import matplotlib.pyplot as plt
from decay_function_base import DecayFunctionBase



class ExponentialDecayFunction(DecayFunctionBase):
  """
  Implementation of exponential decay.
  f(t) = exp(- lambda * t)
  lambda is the decay constant. The time constant is 1 / lambda
  """


  def __init__(self, time_constant=10.0):
    """
    @param (float) lambda_constant: positive exponential decay time constant.
    """
    assert not time_constant < 0
    self._lambda_constant = 1/float(time_constant)


  def decay(self, activationLevel):
    """
    @param activationLevel: current activation level
    @return: activation level after decay
    """
    activationLevel -= self._lambda_constant * activationLevel
    return activationLevel

  def plot(self):
    initValue = 20
    nStep = 20
    x = numpy.arange(0, nStep, 1)
    y = numpy.zeros(x.shape)
    y[0] = initValue

    for i in range(0, nStep-1):
      y[i+1] = self.decay(y[i])

    plt.ion()
    plt.show()

    plt.plot(x, y)
    plt.xlabel('Time after activation (step)')
    plt.ylabel('Persistence')