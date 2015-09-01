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

import random
import csv
import os

import numpy

from nupic.data.fieldmeta import FieldMetaType
from nupic.encoders.base import Encoder, EncoderResult



class LetterEncoder(Encoder):
  """
  A letter encoder encodes an English character into an array of bits.

  It encodes the letter phonologically.  This means that letters that are
  pronounced similarly have similar encodings.  It encodes the multiple ways a
  letter can be pronounced.

  """


  def __init__(self, name="letter", verbosity=0):
    """
    name -- an optional string which will become part of the description

    See class documentation for more information.
    """
    self.w = 24
    self.numGroups = 24
    self.n = self.w * self.numGroups / 2
    self.name = name
    self.encoders = None

    # Each group is associated with a certain set of bits
    groupIndices = {i: range(i * self.w / 2, i * self.w / 2 + (self.w / 2))
      for i in range(self.numGroups)}
    self.letterToBitmap = {}

    # TODO: when selecting bits for things like "th" ensure they are the same bits
    with open(os.path.dirname(__file__) + '/data/letter_mappings.csv') as csvfile:
      reader = csv.reader(csvfile)
      character = 'a'
      for row in reader:
        validGroups = [i for i,r in enumerate(row) if r !=""]
        possibleBits = [i for group in validGroups for i in groupIndices[group]]
        random.seed(character)
        onBits = random.sample(possibleBits, self.w)
        self.letterToBitmap[character] = onBits
        character = chr(ord(character) + 1)


  def getDecoderOutputFieldTypes(self):
    """ [Encoder class virtual method override]
    """
    return (FieldMetaType.string, )


  def getWidth(self):
    """ [Encoder class virtual method override]
    """
    return self.n


  def getDescription(self):
    """ See the function description in base.py """
    return [(self.name, 0)]


  def getBucketIndices(self, inputData):
    """ See method description in base.py """
    for i in range(26):
      if chr(ord("a") + i) == inputData:
        return [i]


  def encodeIntoArray(self, inputData, output, learn=True):
    """ See method description in base.py """
    output[:self.n] = 0
    output[self.letterToBitmap[inputData.lower()]] = 1


  def decode(self, encoded, parentFieldName=''):
    """ See the function description in base.py
    """
    onBits = numpy.where(encoded)
    for letter, bitmap in self.letterToBitmap.iteritems():
      if onBits == bitmap:
        return letter


  def getScalars(self, inputData):
    """ See the function description in base.py """
    return self.getBucketIndices(inputData)[0]

  
  def getEncodedValues(self, inputData):
    """ See the function description in base.py """
    return tuple(inputData)


  def getBucketValues(self):
    """ See the function description in base.py """
    return [chr(ord("a") + i) for i in range(26)]


  def getBucketInfo(self, buckets):
    """ See the function description in base.py """
    # The "category" is simply the bucket index
    category = buckets[0]
    encoding = numpy.zeros(self.n)
    encoding[self.letterToBitmap[category]] = 1

    return [EncoderResult(value=category, scalar=ord(category), encoding=encoding)]


  def topDownCompute(self, encoded):
    """ See the function description in base.py
    """
    character = self.decode(encoded)

    # Return that bucket info
    return self.getBucketInfo([character])


  def closenessScores(self, expValues, actValues):
    """ See the function description in base.py
    """
    expOnBits = numpy.array(self.letterToBitmap[expValues[0]])
    actOnBits = numpy.array(self.letterToBitmap[actValues[0]])

    return numpy.intersect1d(expOnBits, actOnBits) / numpy.union1d(expOnBits, actOnBits)
