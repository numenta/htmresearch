'''DenseNet in PyTorch.'''
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from htmresearch.frameworks.pytorch.linear_sdr import LinearSDR
from htmresearch.frameworks.pytorch.k_winners import (
  KWinnersCNN, updateDutyCycleCNN
)


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
        self.growth_rate = growth_rate

        num_planes = 2*growth_rate
        self.conv1 = nn.Conv2d(3, num_planes, kernel_size=3, padding=1, bias=False)

        self.dense1 = self._make_dense_layers(block, num_planes, nblocks[0])
        num_planes += nblocks[0]*growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        self.trans1 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense2 = self._make_dense_layers(block, num_planes, nblocks[1])
        num_planes += nblocks[1]*growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        self.trans2 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense3 = self._make_dense_layers(block, num_planes, nblocks[2])
        num_planes += nblocks[2]*growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        self.trans3 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense4 = self._make_dense_layers(block, num_planes, nblocks[3])
        num_planes += nblocks[3]*growth_rate

        self.bn = nn.BatchNorm2d(num_planes)
        self.linear = nn.Linear(num_planes, num_classes)

    def _make_dense_layers(self, block, in_planes, nblock):
        layers = []
        for i in range(nblock):
            layers.append(block(in_planes, self.growth_rate))
            in_planes += self.growth_rate
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.trans1(self.dense1(out))
        out = self.trans2(self.dense2(out))
        out = self.trans3(self.dense3(out))
        out = self.dense4(out)
        out = F.avg_pool2d(F.relu(self.bn(out)), 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

    def postEpoch(self):
        print("Null post epoch")


class SparseTransition(nn.Module):
    def __init__(self, in_planes, out_planes, imSize=32*32, sparsity=0.1):
        super(SparseTransition, self).__init__()
        self.bn = nn.BatchNorm2d(in_planes)
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False)
        self.register_buffer("dutyCycle", torch.zeros((1, in_planes, 1, 1)))
        self.dutyCyclePeriod = 1000
        self.boostStrength = 1.5
        self.learningIterations = 0
        self.k = int(sparsity*(in_planes*imSize))
        print "Sparse Transition init: in_planes:", in_planes, "out_planes:", out_planes, "k:", self.k


    def forward(self, x):
        self.learningIterations += x.shape[0]
        out = self.bn(x)
        # print "Transition:", x.shape, "out after BN", out.shape,
        out = KWinnersCNN.apply(out, self.dutyCycle, self.k, self.boostStrength)

        # Update moving average of duty cycle for training iterations only
        # During inference this is kept static.
        updateDutyCycleCNN(out, self.dutyCycle,
                           self.dutyCyclePeriod, self.learningIterations)
        # out = F.relu(out)
        # print "number of non-zeros after k-winners", out[0].nonzero().size(0),
        out = self.conv(out)
        # print "out after conv 1x1", out.shape
        out = F.avg_pool2d(out, 2)
        return out


class SparseDenseNet(nn.Module):
    def __init__(self, block, nblocks, growth_rate=12, reduction=0.5, num_classes=10,
                 sparsity=0.1):
        super(SparseDenseNet, self).__init__()
        self.growth_rate = growth_rate

        num_planes = 2*growth_rate
        self.conv1 = nn.Conv2d(3, num_planes, kernel_size=3, padding=1, bias=False)

        self.dense1 = self._make_dense_layers(block, num_planes, nblocks[0])
        num_planes += nblocks[0]*growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        self.trans1 = SparseTransition(num_planes, out_planes, sparsity=sparsity)
        num_planes = out_planes

        self.dense2 = self._make_dense_layers(block, num_planes, nblocks[1])
        num_planes += nblocks[1]*growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        self.trans2 = SparseTransition(num_planes, out_planes, 16*16, sparsity=sparsity)
        num_planes = out_planes

        self.dense3 = self._make_dense_layers(block, num_planes, nblocks[2])
        num_planes += nblocks[2]*growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        self.trans3 = SparseTransition(num_planes, out_planes, 8*8, sparsity=sparsity)
        num_planes = out_planes

        self.dense4 = self._make_dense_layers(block, num_planes, nblocks[3])
        num_planes += nblocks[3]*growth_rate

        print "Number of inputs into linearSDR=", num_planes
        self.bn = nn.BatchNorm2d(num_planes)
        print "linearSDR weightSparsity = 0.3, k=50/500"
        self.linearSDR = LinearSDR(inputFeatures=num_planes,
                                    n=500,
                                    k=50,
                                    kInferenceFactor=1.5,
                                    weightSparsity=0.3,
                                    boostStrength=1.5,
                                    useBatchNorm=False,
                                    normalizeWeights=False
                                    )

        self.linear = nn.Linear(500, num_classes)

    def _make_dense_layers(self, block, in_planes, nblock):
        layers = []
        for i in range(nblock):
            layers.append(block(in_planes, self.growth_rate))
            in_planes += self.growth_rate
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.trans1(self.dense1(out))
        out = self.trans2(self.dense2(out))
        out = self.trans3(self.dense3(out))
        out = self.dense4(out)
        out = F.avg_pool2d(F.relu(self.bn(out)), 4)
        out = out.view(out.size(0), -1)
        out = self.linearSDR(out)
        out = self.linear(out)
        return out

    def postEpoch(self):
        if self.training:
            self.trans1.boostStrength = self.trans1.boostStrength * 0.9
            self.linearSDR.setBoostStrength(self.linearSDR.boostStrength * 0.9)
            self.linearSDR.rezeroWeights()
            print "boostStrength is now:", self.linearSDR.boostStrength


def DenseNet121():
    return DenseNet(Bottleneck, [6,12,24,16], growth_rate=32)

def SparseDenseNet121():
    return SparseDenseNet(Bottleneck, [6,12,24,16], growth_rate=32)

def DenseNet169():
    return DenseNet(Bottleneck, [6,12,32,32], growth_rate=32)

def DenseNet201():
    return DenseNet(Bottleneck, [6,12,48,32], growth_rate=32)

def DenseNet161():
    return DenseNet(Bottleneck, [6,12,36,24], growth_rate=48)

def densenet_cifar(growth_rate=12):
    print "Running densenet_cifar with growth rate", growth_rate
    return DenseNet(Bottleneck, [6,12,24,16], growth_rate=growth_rate)

def sparse_densenet_cifar(sparsity=0.1, growth_rate=12):
    print "Running sparse_densenet_cifar with sparsity=",sparsity,
    print "growth rate=", growth_rate
    return SparseDenseNet(Bottleneck, [6,12,24,16], growth_rate=growth_rate, sparsity=sparsity)

def test():
    net = densenet_cifar()
    x = torch.randn(1,3,32,32)
    y = net(x)
    print(y)

# test()
