'''DenseNet in PyTorch.'''
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from tabulate import tabulate

from htmresearch.frameworks.pytorch.modules import (
  SparseWeights, KWinners2d, KWinners,
  updateBoostStrength, rezeroWeights)


class Bottleneck(nn.Module):
  def __init__(self, in_planes, growth_rate):
    super(Bottleneck, self).__init__()
    self.bn1 = nn.BatchNorm2d(in_planes)
    self.conv1 = nn.Conv2d(in_planes, 4*growth_rate, kernel_size=1, bias=False)
    self.bn2 = nn.BatchNorm2d(4*growth_rate)
    self.conv2 = nn.Conv2d(4*growth_rate, growth_rate, kernel_size=3, padding=1, bias=False)

  def forward(self, x):
    out = self.conv1(F.relu(self.bn1(x)))
    out = self.conv2(F.relu(self.bn2(out)))
    out = torch.cat([out,x], 1)
    return out


class Transition(nn.Module):
  def __init__(self, in_planes, out_planes):
    super(Transition, self).__init__()
    self.bn = nn.BatchNorm2d(in_planes)
    self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False)

  def forward(self, x):
    out = self.bn(x)
    out = self.conv(F.relu(out))
    out = F.avg_pool2d(out, 2)
    return out


class DenseNet(nn.Module):
  def __init__(self, block, nblocks, growth_rate=12, reduction=0.5, num_classes=10):
    super(DenseNet, self).__init__()

    self.iteration=0
    self.growth_rate = growth_rate

    print "Creating dense network with nblocks=",nblocks,"and growth_rate=",growth_rate

    num_planes = 2*growth_rate
    self.conv1 = nn.Conv2d(3, num_planes, kernel_size=3, padding=1, bias=False)

    print
    print "Creating dense block 1 with num blocks=",nblocks[0]
    self.dense1 = self._make_dense_layers(block, num_planes, nblocks[0])
    num_planes += nblocks[0]*growth_rate
    out_planes = int(math.floor(num_planes*reduction))
    print "Transition. in_channels:", num_planes, "| out_channels:", out_planes
    self.trans1 = Transition(num_planes, out_planes)
    num_planes = out_planes

    print
    print "Creating dense block 2 with num blocks=",nblocks[1]
    self.dense2 = self._make_dense_layers(block, num_planes, nblocks[1])
    num_planes += nblocks[1]*growth_rate
    out_planes = int(math.floor(num_planes*reduction))
    print "Transition. in_channels:", num_planes, "| out_channels:", out_planes
    self.trans2 = Transition(num_planes, out_planes)
    num_planes = out_planes

    print
    print "Creating dense block 3 with num blocks=",nblocks[2]
    self.dense3 = self._make_dense_layers(block, num_planes, nblocks[2])
    num_planes += nblocks[2]*growth_rate
    out_planes = int(math.floor(num_planes*reduction))
    print "Transition. in_channels:", num_planes, "| out_channels:", out_planes
    self.trans3 = Transition(num_planes, out_planes)
    num_planes = out_planes

    print
    print "Creating dense block 4 with num blocks=",nblocks[3]
    self.dense4 = self._make_dense_layers(block, num_planes, nblocks[3])
    num_planes += nblocks[3]*growth_rate

    self.bn = nn.BatchNorm2d(num_planes)
    self.linear = nn.Linear(num_planes, num_classes)

  def _make_dense_layers(self, block, in_planes, nblock):
    layers = []
    for i in range(nblock):
      layers.append(block(in_planes, self.growth_rate))
      in_planes += self.growth_rate

    for l,layer in enumerate(layers):
      print "Layer:",l,"in_channels for conv1:",layer.conv1.in_channels,
      print "| in_channels for conv2:",layer.conv2.in_channels,"| out_channels for conv2:",layer.conv2.out_channels
    return nn.Sequential(*layers)

  def forward(self, x):
    out1 = self.conv1(x)
    out2 = self.trans1(self.dense1(out1))
    out3 = self.trans2(self.dense2(out2))
    out4 = self.trans3(self.dense3(out3))
    out5 = self.dense4(out4)
    out6 = F.avg_pool2d(F.relu(self.bn(out5)), 4)
    out7 = out6.view(out6.size(0), -1)
    out = self.linear(out7)

    if self.iteration == 0:
      print "Channels and vector size coming out of each stage"
      print "Input image", x.shape[1], x.shape[1]*x.shape[2]*x.shape[3]
      print "First convolution layer", out1.shape[1], out1.shape[1]*out1.shape[2]*out1.shape[3]
      print "First transition layer", out2.shape[1], out2.shape[1]*out2.shape[2]*out2.shape[3]
      print "Second transition layer", out3.shape[1], out3.shape[1]*out3.shape[2]*out3.shape[3]
      print "Third transition layer", out4.shape[1], out4.shape[1]*out4.shape[2]*out4.shape[3]
      print "Fourth dense block ", out5.shape[1], out5.shape[1]*out5.shape[2]*out5.shape[3]
      print "Average pool", out6.shape[1]
      print "Into output layer", out7.shape[1]

    self.iteration += 1
    return out

  def postEpoch(self):
    print("Null post epoch")


#####################################################################


class SparseBottleneck(nn.Module):
  def __init__(self, in_planes, growth_rate, input_width, sparsity=0.1):
    super(SparseBottleneck, self).__init__()
    self.bn1 = nn.BatchNorm2d(in_planes)
    self.conv1 = nn.Conv2d(in_planes, 4*growth_rate, kernel_size=1, bias=False)
    self.bn2 = nn.BatchNorm2d(4*growth_rate)
    self.conv2 = nn.Conv2d(4*growth_rate, growth_rate, kernel_size=3, padding=1, bias=False)

    conv2OutputSize = (growth_rate * input_width * input_width)
    self.k = int(sparsity * conv2OutputSize)
    if sparsity < 0.5:
      self.kwinners2 = KWinners2d(
        n=conv2OutputSize, k=self.k, channels=growth_rate,
        kInferenceFactor=1.25, boostStrength=1.5, boostStrengthFactor=0.95)
      print "SparseBottleneck init: in_planes:", in_planes, "conv2OutputSize:", conv2OutputSize, "k:", self.k
    else:
      print "SparseBottleneck init: Skipping k-winners, sparsity too high"
      self.kwinners2 = None

  def forward(self, x):
    out = self.conv1(F.relu(self.bn1(x)))
    out = self.conv2(F.relu(self.bn2(out)))
    if self.kwinners2 is not None:
      out = self.kwinners2(out)
    out = torch.cat([out,x], 1)
    return out




class SparseTransition(nn.Module):
  def __init__(self, in_planes, out_planes, imSize=32*32, sparsity=0.1):
    super(SparseTransition, self).__init__()
    self.bn = nn.BatchNorm2d(in_planes)
    self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False)

    transitionOutputSize = in_planes * imSize / 4
    self.k = int(sparsity * transitionOutputSize)
    if sparsity < 0.5:
      self.kwinners = KWinners2d(
        n=transitionOutputSize, k=self.k, channels=out_planes,
        kInferenceFactor=1.25,
        boostStrength=1.5,
        boostStrengthFactor=0.95)
      print "Sparse Transition init: in_planes:", in_planes, "out_planes:", out_planes, "k:", self.k
    else:
      print "Sparse Transition init: Skipping k-winners, sparsity too high"
      self.kwinners = None


  def forward(self, x):
    out = self.bn(x)
    out = self.conv(out)
    out = F.avg_pool2d(out, 2)
    if self.kwinners is not None:
      out = self.kwinners(out)
    return out


class NotSoDenseNet(nn.Module):
  def __init__(self, block, nblocks, growth_rate=12, reduction=0.5, num_classes=10,
               dense_sparsities=[0.1, 0.1, 0.2, 0.5],
               transition_sparsities=[0.1, 0.1, 0.1],
               linear_sparsity=0.1,
               linear_weight_sparsity=0.3,
               image_width=32):
    super(NotSoDenseNet, self).__init__()
    self.growth_rate = growth_rate
    self.iteration = 0
    self.linear_sparsity = linear_sparsity

    print "Creating Sparse network with nblocks=",nblocks,"and growth_rate=",growth_rate
    print "dense_sparsities=", dense_sparsities
    print "transition_sparsities=",transition_sparsities
    print "linear_sparsity=", linear_sparsity, "linear_weight_sparsity=", linear_weight_sparsity

    num_planes = 2*growth_rate
    print
    print "Creating first CNN layer with out_channels=",num_planes
    self.conv1 = nn.Conv2d(3, num_planes, kernel_size=3, padding=1, bias=False)

    print
    print "Creating dense block 1 with num blocks=",nblocks[0]
    self.dense1 = self._make_dense_layers(block, num_planes,
                                          nblocks[0], image_width,
                                          sparsity=dense_sparsities[0])
    num_planes += nblocks[0]*growth_rate
    out_planes = int(math.floor(num_planes*reduction))
    # self.trans1 = Transition(num_planes, out_planes)
    print "Transition. in_channels:", num_planes, "| out_channels:", out_planes
    self.trans1 = SparseTransition(num_planes, out_planes,
                                   imSize=image_width*image_width,
                                   sparsity=transition_sparsities[0])
    num_planes = out_planes

    print
    print "Creating dense block 2 with num blocks=",nblocks[1]
    self.dense2 = self._make_dense_layers(block, num_planes,
                                          nblocks[1], image_width / 2,
                                          sparsity=dense_sparsities[1])
    num_planes += nblocks[1]*growth_rate
    out_planes = int(math.floor(num_planes*reduction))
    # self.trans2 = Transition(num_planes, out_planes)
    print "Transition. in_channels:", num_planes, "| out_channels:", out_planes
    self.trans2 = SparseTransition(num_planes, out_planes,
                                   imSize=16 * 16,
                                   sparsity=transition_sparsities[1])
    num_planes = out_planes

    print
    print "Creating dense block 3 with num blocks=",nblocks[2]
    self.dense3 = self._make_dense_layers(block, num_planes,
                                          nblocks[2], image_width / 4,
                                          sparsity=dense_sparsities[2])
    num_planes += nblocks[2]*growth_rate
    out_planes = int(math.floor(num_planes*reduction))
    # self.trans3 = Transition(num_planes, out_planes)
    print "Transition. in_channels:", num_planes, "| out_channels:", out_planes
    self.trans3 = SparseTransition(num_planes, out_planes,
                                   imSize=8 * 8,
                                   sparsity=transition_sparsities[2])
    num_planes = out_planes

    print
    print "Creating dense block 4 with num blocks=",nblocks[3]
    self.dense4 = self._make_dense_layers(block, num_planes,
                                          nblocks[3], image_width / 8,
                                          sparsity=dense_sparsities[3])
    num_planes += nblocks[3]*growth_rate

    self.bn = nn.BatchNorm2d(num_planes)

    if self.linear_sparsity > 0:
      print "Number of inputs into linearSDR=", num_planes
      print "linearSDR weightSparsity = 0.3, k=50/500"
      self.linear1 = SparseWeights(nn.Linear(num_planes, 500),
                                   weightSparsity=linear_weight_sparsity)
      k = int(500*linear_sparsity)
      self.linear1KWinners = KWinners(
        n=500, k=k, kInferenceFactor=1.5,
        boostStrength=1.5,
        boostStrengthFactor=0.95)
      self.linearOut = nn.Linear(500, num_classes)
    else:
      print "Skipping linear sparse layer"
      self.linear1KWinners = None
      self.linearOut = nn.Linear(num_planes, num_classes)


  def _make_dense_layers(self, block, in_planes, nblock, input_width, sparsity):
    layers = []
    for i in range(nblock):
      layers.append(block(in_planes, self.growth_rate, input_width, sparsity=sparsity))
      in_planes += self.growth_rate

    for l,layer in enumerate(layers):
      print "Layer:",l,"in_channels for conv1:",layer.conv1.in_channels,
      print "| in_channels for conv2:", layer.conv2.in_channels, "| out_channels for conv2:", layer.conv2.out_channels

    return nn.Sequential(*layers)


  def forwardWithTable(self, x):
    paramsTable = [["Name", "out_channels", "Output Size", "k", "Non-zeros"]]

    paramsTable.append(["Image", x.shape[1], x.shape[1]*x.shape[2]*x.shape[3],
                        0, x[0].nonzero().size(0)])

    out = self.conv1(x)
    paramsTable.append(["Conv1", out.shape[1], out.shape[1]*out.shape[2]*out.shape[3],
                        0, out[0].nonzero().size(0)])

    out = self.dense1(out)
    paramsTable.append(["Dense1", out.shape[1], out.shape[1]*out.shape[2]*out.shape[3],
                        0, out[0].nonzero().size(0)])

    out = self.trans1(out)
    paramsTable.append(["Trans1", out.shape[1], out.shape[1]*out.shape[2]*out.shape[3],
                        0, out[0].nonzero().size(0)])

    out = self.dense2(out)
    paramsTable.append(["Dense2", out.shape[1], out.shape[1]*out.shape[2]*out.shape[3],
                        0, out[0].nonzero().size(0)])

    out = self.trans2(out)
    paramsTable.append(["Trans2", out.shape[1], out.shape[1]*out.shape[2]*out.shape[3],
                        0, out[0].nonzero().size(0)])

    out = self.dense3(out)
    paramsTable.append(["Dense3", out.shape[1], out.shape[1]*out.shape[2]*out.shape[3],
                        0, out[0].nonzero().size(0)])

    out = self.trans3(out)
    paramsTable.append(["Trans3", out.shape[1], out.shape[1]*out.shape[2]*out.shape[3],
                        0, out[0].nonzero().size(0)])

    out = self.dense4(out)
    paramsTable.append(["Dense4", out.shape[1], out.shape[1]*out.shape[2]*out.shape[3],
                        0, out[0].nonzero().size(0)])

    out = F.avg_pool2d(F.relu(self.bn(out)), 4)
    paramsTable.append(["AvgPool", out.shape[1], out.shape[1]*out.shape[2]*out.shape[3],
                        0, out[0].nonzero().size(0)])

    out = out.view(out.size(0), -1)

    if self.linear_sparsity > 0:
      out = self.linear1KWinners(self.linear1(out))
      paramsTable.append(["LinearSDR", 1, out.shape[1],
                          0, out[0].nonzero().size(0)])

    out = self.linearOut(out)

    print ""
    print tabulate(paramsTable, headers="firstrow", tablefmt="grid")

    return out


  def forward(self, x):
    self.iteration += 1
    if self.iteration == 1:
      return self.forwardWithTable(x)

    out = self.conv1(x)
    out = self.trans1(self.dense1(out))
    out = self.trans2(self.dense2(out))
    out = self.trans3(self.dense3(out))
    out = self.dense4(out)
    out = F.avg_pool2d(F.relu(self.bn(out)), 4)
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


def DenseNet121():
  return DenseNet(Bottleneck, [6,12,24,16], growth_rate=32)

def SparseDenseNet121():
  return NotSoDenseNet(Bottleneck, [6, 12, 24, 16], growth_rate=32)

def DenseNet169():
  return DenseNet(Bottleneck, [6,12,32,32], growth_rate=32)

def DenseNet201():
  return DenseNet(Bottleneck, [6,12,48,32], growth_rate=32)

def DenseNet161():
  return DenseNet(Bottleneck, [6,12,36,24], growth_rate=48)

# def densenet_cifar(growth_rate=12):
#   print "Running densenet_cifar with growth rate", growth_rate
#   return DenseNet(Bottleneck, [6,12,24,16], growth_rate=growth_rate)

def densenet_cifar(growth_rate=12):
  print "Running densenet_cifar with growth rate", growth_rate
  # return DenseNet(Bottleneck, [6,12,24,16], growth_rate=growth_rate)
  return NotSoDenseNet(SparseBottleneck, [6, 12, 24, 16],
                       growth_rate=growth_rate,
                       dense_sparsities=[1.0]*4,
                       transition_sparsities=[1.0]*3,
                       linear_sparsity=0.0
                       )

def notso_densenet_cifar(growth_rate=12):
  print "Running notso_densenet_cifar with growth rate=", growth_rate
  return NotSoDenseNet(SparseBottleneck, [6, 12, 24, 16],
                       growth_rate=growth_rate,
                       dense_sparsities=[0.2, 1.0, 1.0, 1.0],
                       transition_sparsities=[0.1, 0.1, 0.2],
                       linear_sparsity=0.0,
                       linear_weight_sparsity=0.3,
                       )

def test():
  net = densenet_cifar()
  x = torch.randn(1,3,32,32)
  y = net(x)
  print(y)

# test()
