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
from torch.autograd import Variable

class KWinners(torch.autograd.Function):
  """
  A simplistic K-winner take all autograd function for experimenting with
  sparsity (k currently hardcoded!)

  Code adapted from the excellent tutorial:
  https://github.com/jcjohnson/pytorch-examples
  """
  @staticmethod
  def forward(ctx, x, dutyCycles, k, boostStrength):
    """
    In the forward pass we receive a context object and a Tensor containing the
    input; we must return a Tensor containing the output, and we can use the
    context object to cache objects for use in the backward pass.
    """
    targetDensity = float(k) / x.shape[1]
    boostFactors = torch.exp((targetDensity - dutyCycles) * boostStrength)
    boosted = x * boostFactors
    # print(boosted)
    res = torch.zeros(x.shape)
    topk, indices = boosted.topk(k)

    # res = res.scatter(1, indices, topk)
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
    grad_x = Variable(torch.zeros(grad_output.shape))

    # Probably a better way to do it, but this is not terrible as it loops
    # over the batch size.
    for i in range(grad_output.size(0)):
      grad_x[i, indices[i]] = grad_output[i, indices[i]]

    return grad_x,None,None,None

