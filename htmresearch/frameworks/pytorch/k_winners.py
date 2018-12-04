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
import torch


class KWinners(torch.autograd.Function):
  """
  A simplistic K-winner take all autograd function for experimenting with
  sparsity.

  Code adapted from this excellent tutorial:
  https://github.com/jcjohnson/pytorch-examples
  """

  @staticmethod
  def forward(ctx, x, dutyCycles, k, boostStrength):
    """
    In the forward pass we receive a context object and a Tensor containing the
    input; we must return a Tensor containing the output, and we can use the
    context object to cache objects for use in the backward pass.

    Use the boost strength to compute a boost factor for each unit represented
    in x. These factors are used to increase the impact of each unit to improve
    their chances of being chosen. This encourages participation of more columns
    in the learning process.

    The boosting function is a curve defined as: boostFactors = exp[ -
    boostStrength * (dutyCycle - targetDensity)] Intuitively this means that
    units that have been active (i.e. in the top-k) at the target activation
    level have a boost factor of 1, meaning their activity is not boosted.
    Columns whose duty cycle drops too much below that of their neighbors are
    boosted depending on how infrequently they have been active. Unit that has
    been active more than the target activation level have a boost factor below
    1, meaning their activity is suppressed and they are less likely to be in 
    the top-k.

    Note that we do not transmit the boosted values. We only use boosting to
    determine the winning units.

    The target activation density for each unit is k / number of units. The
    boostFactor depends on the dutyCycle via an exponential function:

            boostFactor
                ^
                |
                |\
                | \
          1  _  |  \
                |    _
                |      _ _
                |          _ _ _ _
                +--------------------> dutyCycle
                   |
              targetDensity

    :param ctx: 
      Place where we can store information we will need to compute the gradients
      for the backward pass.

    :param x: 
      Current activity of each unit.  

    :param dutyCycles: 
      The averaged duty cycle of each unit.

    :param k: 
      The activity of the top k units will be allowed to remain, the rest are
      set to zero.
                
    :param boostStrength:     
      A boost strength of 0.0 has no effect on x.

    :return: 
      A tensor representing the activity of x after k-winner take all.
    """
    if boostStrength > 0.0:
      targetDensity = float(k) / x.shape[1]
      boostFactors = torch.exp((targetDensity - dutyCycles) * boostStrength)
      boosted = x.detach() * boostFactors
    else:
      boosted = x.detach()

    # Take the boosted version of the input x, find the top k winners.
    # Compute an output that contains the values of x corresponding to the top k
    # boosted values
    res = torch.zeros_like(x)
    topk, indices = boosted.topk(k, sorted=False)
    for i in range(x.shape[0]):
      res[i,indices[i]] = x[i,indices[i]]

    ctx.save_for_backward(x,indices)
    return res


  @staticmethod
  def backward(ctx, grad_output):
    """
    In the backward pass we receive the context object and a Tensor containing
    the gradient of the loss with respect to the output produced during the
    forward pass. We can retrieve cached data from the context object, and must
    compute and return the gradient of the loss with respect to the input to the
    forward function.
    """
    x,indices, = ctx.saved_tensors
    grad_x = torch.zeros_like(grad_output, requires_grad=True)

    # Probably a better way to do it, but this is not terrible as it only loops
    # over the batch size.
    for i in range(grad_output.size(0)):
      grad_x[i, indices[i]] = grad_output[i, indices[i]]

    return grad_x, None, None, None

