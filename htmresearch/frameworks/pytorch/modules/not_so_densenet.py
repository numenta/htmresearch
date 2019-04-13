# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2019, Numenta, Inc.  Unless you have an agreement
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
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from tabulate import tabulate

from htmresearch.frameworks.pytorch.modules import (
  SparseWeights, KWinners2d, KWinners, updateBoostStrength, rezeroWeights
)



class SparseBlock(nn.Module):
  def __init__(self, in_planes,
               input_width,
               c1_out_planes,
               c2_out_planes,
               sparsity=0.1,
               k_inference_factor=1.5):
    super(SparseBlock, self).__init__()
    self.bn1 = nn.BatchNorm2d(in_planes)
    self.conv1 = nn.Conv2d(in_planes, c1_out_planes, kernel_size=1, bias=False)
    self.bn2 = nn.BatchNorm2d(c1_out_planes)
    self.conv2 = nn.Conv2d(c1_out_planes, c2_out_planes, kernel_size=3, padding=1, bias=False)
    self.iterations = 0

    conv2OutputSize = (c2_out_planes * input_width * input_width)
    self.k = int(sparsity * conv2OutputSize)
    if sparsity < 0.5:
      self.kwinners2 = KWinners2d(
        n=conv2OutputSize, k=self.k, channels=c2_out_planes,
        kInferenceFactor=k_inference_factor,
        boostStrength=1.5, boostStrengthFactor=0.95)
    else:
      self.kwinners2 = None


  def forward(self, x):
    out = self.conv1(F.relu(self.bn1(x)))
    c1s = out.shape[1:]
    out = self.conv2(F.relu(self.bn2(out)))
    c2s = out.shape[1:]
    if self.kwinners2 is not None:
      out = self.kwinners2(out)
    out = torch.cat([out,x], 1)

    if self.iterations == 0:
      print "SparseBottleneck forward: "
      print "           input shape", x.shape[1:], "size:", np.prod(x.shape[1:])
      print "    conv1 output shape", c1s, "size:", np.prod(c1s), "weight size", x.shape[1]
      print "    conv2 output shape", c2s, "size:", np.prod(c2s), "weight size", c1s[0]*np.prod(self.conv2.kernel_size)
      print "    final output shape", out.shape[1:], "size:", np.prod(out.shape[1:])

    self.iterations += 1
    return out


class SparseTransition(nn.Module):
  def __init__(self, in_planes, out_planes, imSize=32*32,
               sparsity=0.1,
               k_inference_factor=1.5):
    super(SparseTransition, self).__init__()
    self.bn = nn.BatchNorm2d(in_planes)
    self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False)
    self.iterations = 0

    transitionOutputSize = out_planes * imSize / 4
    self.k = int(sparsity * transitionOutputSize)
    if sparsity < 0.5:
      self.kwinners = KWinners2d(
        n=transitionOutputSize, k=self.k, channels=out_planes,
        kInferenceFactor=k_inference_factor,
        boostStrength=1.5,
        boostStrengthFactor=0.95)
    else:
      self.kwinners = None


  def forward(self, x):
    out = self.bn(x)
    out = self.conv(out)
    c1s = out.shape[1:]
    out = F.avg_pool2d(out, 2)

    if self.iterations == 0:
      print "SparseTransition forward: "
      print "              input shape", x.shape[1:], "size:", np.prod(x.shape[1:])
      print "        conv output shape", c1s, "size:", np.prod(c1s)
      print "    avg_pool output shape", out.shape[1:], "size:", np.prod(out.shape[1:])
      print ""

    if self.kwinners is not None:
      out = self.kwinners(out)

    self.iterations += 1

    return out


class NotSoDenseNet(nn.Module):
  def __init__(self, nblocks,
               growth_rate=12,
               reduction=0.5,
               num_classes=10,
               block=SparseBlock,
               conv1_sparsity = 1.0,
               dense_c1_out_planes=4*12,
               dense_sparsities=[0.1, 0.1, 0.2, 0.5],
               transition_sparsities=[0.1, 0.1, 0.1],
               linear_sparsity=0.1,
               linear_weight_sparsity=0.3,
               linear_n=500,
               avg_pool_size=4,
               k_inference_factor=1.5,
               image_width=32):
    super(NotSoDenseNet, self).__init__()
    self.growth_rate = growth_rate
    self.iteration = 0
    self.linear_sparsity = linear_sparsity
    self.avg_pool_size = avg_pool_size
    self.dense_sparsities = dense_sparsities
    self.k_inference_factor = k_inference_factor

    print "Creating NotSoDenseNets with nblocks=", nblocks
    print "growth_rate=",growth_rate
    print "dense_c1_out_planes=", dense_c1_out_planes
    print "dense_sparsities=", dense_sparsities
    print "transition_sparsities=", transition_sparsities
    print "k_inference_factor=", k_inference_factor
    print "linear_sparsity=", linear_sparsity, "linear_weight_sparsity=", linear_weight_sparsity

    num_planes = 2*growth_rate
    self.conv1 = nn.Conv2d(3, num_planes, kernel_size=3, padding=1, bias=False)
    conv1OutputSize = image_width*image_width*num_planes
    self.conv1k = int(conv1_sparsity * conv1OutputSize)
    self.conv1Sparsity = conv1_sparsity
    if self.conv1Sparsity < 0.5:
      self.conv1kwinners = KWinners2d(
        n=conv1OutputSize, k=self.conv1k, channels=num_planes,
        kInferenceFactor=k_inference_factor,
        boostStrength=1.5, boostStrengthFactor=0.95)

    self.dense1 = self._make_sparse_blocks(block, num_planes,
                                           dense_c1_out_planes,
                                           nblocks[0], image_width,
                                           sparsity=dense_sparsities[0])
    num_planes += nblocks[0]*growth_rate
    out_planes = int(math.floor(num_planes*reduction))
    self.trans1 = SparseTransition(num_planes, out_planes,
                                   imSize=image_width*image_width,
                                   sparsity=transition_sparsities[0],
                                   k_inference_factor=self.k_inference_factor,
                                   )
    num_planes = out_planes

    self.dense2 = self._make_sparse_blocks(block, num_planes,
                                           dense_c1_out_planes,
                                           nblocks[1], image_width / 2,
                                           sparsity=dense_sparsities[1])
    num_planes += nblocks[1]*growth_rate
    out_planes = int(math.floor(num_planes*reduction))
    self.trans2 = SparseTransition(num_planes, out_planes,
                                   imSize=16 * 16,
                                   sparsity=transition_sparsities[1],
                                   k_inference_factor=self.k_inference_factor,
                                   )
    num_planes = out_planes

    self.dense3 = self._make_sparse_blocks(block, num_planes,
                                           dense_c1_out_planes,
                                           nblocks[2], image_width / 4,
                                           sparsity=dense_sparsities[2],
                                           )
    num_planes += nblocks[2]*growth_rate
    out_planes = int(math.floor(num_planes*reduction))
    self.trans3 = SparseTransition(num_planes, out_planes,
                                   imSize=8 * 8,
                                   sparsity=transition_sparsities[2],
                                   k_inference_factor=self.k_inference_factor,
                                   )
    num_planes = out_planes

    self.dense4 = self._make_sparse_blocks(block, num_planes,
                                           dense_c1_out_planes,
                                           nblocks[3], image_width / 8,
                                           sparsity=dense_sparsities[3])
    num_planes += nblocks[3]*growth_rate

    self.bn = nn.BatchNorm2d(num_planes)

    bn_outputs = int(num_planes * 16 / (self.avg_pool_size*self.avg_pool_size))

    if self.linear_sparsity > 0:
      self.linear1 = SparseWeights(nn.Linear(bn_outputs, linear_n),
                                   weightSparsity=linear_weight_sparsity)
      k = int(linear_n*linear_sparsity)
      self.linear1KWinners = KWinners(
        n=linear_n, k=k, kInferenceFactor=k_inference_factor,
        boostStrength=1.5,
        boostStrengthFactor=0.95)
      self.linearOut = nn.Linear(linear_n, num_classes)
    else:
      self.linear1KWinners = None
      self.linearOut = nn.Linear(bn_outputs, num_classes)


  def _make_sparse_blocks(self, block, in_planes,
                          c1_out_planes,
                          nblock, input_width, sparsity):
    layers = []
    for i in range(nblock):
      layers.append(block(in_planes,
                          c1_out_planes=c1_out_planes,
                          c2_out_planes=self.growth_rate,
                          input_width=input_width,
                          sparsity=sparsity,
                          k_inference_factor=self.k_inference_factor
                          ))
      in_planes += self.growth_rate

    return nn.Sequential(*layers)


  def forwardWithTable(self, x):
    paramsTable = [["Name", "out_channels", "Output Size", "k", "Non-zeros"]]

    paramsTable.append(["Image", x.shape[1], x.shape[1]*x.shape[2]*x.shape[3],
                        0, x[0].nonzero().size(0)])

    out = self.conv1(x)
    if self.conv1Sparsity < 0.5:
      out = self.conv1kwinners(out)
    paramsTable.append(["Conv1", out.shape[1], np.prod(out.shape[1:]),
                        self.conv1k, out[0].nonzero().size(0)])

    out = self.dense1(out)
    paramsTable.append(["Dense1", out.shape[1], np.prod(out.shape[1:]),
                        self.dense1[-1].k, out[0].nonzero().size(0)])

    out = self.trans1(out)
    paramsTable.append(["Trans1", out.shape[1], np.prod(out.shape[1:]),
                        self.trans1.k, out[0].nonzero().size(0)])

    out = self.dense2(out)
    paramsTable.append(["Dense2", out.shape[1], np.prod(out.shape[1:]),
                        self.dense2[-1].k, out[0].nonzero().size(0)])

    out = self.trans2(out)
    paramsTable.append(["Trans2", out.shape[1], np.prod(out.shape[1:]),
                        self.trans2.k, out[0].nonzero().size(0)])

    out = self.dense3(out)
    paramsTable.append(["Dense3", out.shape[1], np.prod(out.shape[1:]),
                        self.dense3[-1].k, out[0].nonzero().size(0)])

    out = self.trans3(out)
    paramsTable.append(["Trans3", out.shape[1], np.prod(out.shape[1:]),
                        self.trans3.k, out[0].nonzero().size(0)])

    out = self.dense4(out)
    paramsTable.append(["Dense4", out.shape[1], np.prod(out.shape[1:]),
                        self.dense4[-1].k, out[0].nonzero().size(0)])

    out = F.avg_pool2d(F.relu(self.bn(out)), self.avg_pool_size)
    paramsTable.append(["AvgPool", out.shape[1], np.prod(out.shape[1:]),
                        0, out[0].nonzero().size(0)])

    out = out.view(out.size(0), -1)

    if self.linear_sparsity > 0:
      out = self.linear1KWinners(self.linear1(out))
      paramsTable.append(["LinearSDR", 1, out.shape[1],
                          self.linear1KWinners.k, out[0].nonzero().size(0)])

    out = self.linearOut(out)

    print ""
    print tabulate(paramsTable, headers="firstrow", tablefmt="grid")

    return out


  def forward(self, x):
    self.iteration += 1
    if self.iteration == 1:
      return self.forwardWithTable(x)

    out = self.conv1(x)
    if self.conv1Sparsity < 0.5:
      out = self.conv1kwinners(out)

    out = self.trans1(self.dense1(out))
    out = self.trans2(self.dense2(out))
    out = self.trans3(self.dense3(out))
    out = self.dense4(out)
    out = F.avg_pool2d(F.relu(self.bn(out)), self.avg_pool_size)
    out = out.view(out.size(0), -1)
    if self.linear_sparsity > 0:
      out = self.linear1KWinners(self.linear1(out))

    out = self.linearOut(out)

    return out


  def postEpoch(self):
    self.apply(updateBoostStrength)
    self.apply(rezeroWeights)
    if self.linear1KWinners is not None:
      print "boostStrength is now:", self.linear1KWinners.boostStrength



# def DenseNet121():
#   return DenseNet(Bottleneck, [6,12,24,16], growth_rate=32)
#
# def SparseDenseNet121():
#   return NotSoDenseNet(Bottleneck, [6, 12, 24, 16], growth_rate=32)
#
# def DenseNet169():
#   return DenseNet(Bottleneck, [6,12,32,32], growth_rate=32)
#
# def DenseNet201():
#   return DenseNet(Bottleneck, [6,12,48,32], growth_rate=32)
#
# def DenseNet161():
#   return DenseNet(Bottleneck, [6,12,36,24], growth_rate=48)

