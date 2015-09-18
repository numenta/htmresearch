#!/usr/bin/env python
# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2015, Numenta, Inc.  Unless you have purchased from
# Numenta, Inc. a separate commercial license for this software code, the
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

"""Tests for data_split module."""

import unittest

from htmresearch.support import data_split



class DataSplitTest(unittest.TestCase):


  def testKFoldsBadKType(self):
    with self.assertRaises(TypeError):
      data_split.KFolds(3.0)


  def testKFoldsBadKValue(self):
    with self.assertRaises(ValueError):
      data_split.KFolds(1)


  def testKFolds2(self):
    kfolds = data_split.KFolds(2)

    with self.assertRaises(ValueError):
      kfolds.split(xrange(1))

    self.assertSequenceEqual(
        kfolds.split(xrange(2)),
        [([1], [0]), ([0], [1])])

    self.assertSequenceEqual(
        kfolds.split(xrange(3)),
        [([1, 2], [0]), ([0, 2], [1])])


  def testKFolds3(self):
    kfolds = data_split.KFolds(3)

    with self.assertRaises(ValueError):
      kfolds.split(xrange(2))

    self.assertSequenceEqual(
        kfolds.split(xrange(3)),
        [([1, 2], [0]), ([0, 2], [1]), ([0, 1], [2])])

    self.assertSequenceEqual(
        kfolds.split(xrange(4)),
        [([1, 2, 3], [0]), ([0, 2, 3], [1]), ([0, 1, 3], [2])])


  def testKFoldsRandomize(self):
    kfolds = data_split.KFolds(2)
  
    partition1 = kfolds.split(range(10))
    partition2 = kfolds.split(range(10), randomize=True)
  
    self.assertNotEqual(partition1, partition2)
    self.assertItemsEqual(
      partition1[0][0]+partition1[0][1], partition2[0][0]+partition2[0][1])


  def testStandardSplitBadValue(self):
    with self.assertRaises(ValueError):
      data_split.StandardSplit(2)


  def testStandardSplitNoData(self):
    splitter = data_split.StandardSplit()
    
    with self.assertRaises(ValueError):
      splitter.split([])


  def testStandardSplit(self):
    splitter = data_split.StandardSplit()

    self.assertSequenceEqual(
      splitter.split(xrange(10)),
      ([0, 1, 2, 3, 4, 5, 6, 7], [8, 9]))

    self.assertSequenceEqual(
      splitter.split(xrange(11)),
      ([0, 1, 2, 3, 4, 5, 6, 7], [8, 9, 10]))

    splitter = data_split.StandardSplit(0.7)

    self.assertSequenceEqual(
      splitter.split(xrange(10)),
      ([0, 1, 2, 3, 4, 5, 6], [7, 8, 9]))


  def testStandardSplitRandomize(self):
    splitter = data_split.StandardSplit()

    partition1 = splitter.split(range(10))
    partition2 = splitter.split(range(10), randomize=True)

    self.assertNotEqual(partition1, partition2)
    self.assertItemsEqual(
      partition1[0]+partition1[1], partition2[0]+partition2[1])


if __name__ == "__main__":
  unittest.main()
