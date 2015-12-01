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

import math
import numpy
import random

from nupic.encoders.utils import bitsToString


# Default SDR dimensions for 2% sparsity:
DEFAULT_N = 16384
DEFAULT_W = 328



class LanguageEncoder(object):
  """
  An encoder converts a value to a sparse distributed representation (SDR).

  The Encoder superclass implements:
  - bitmapToSDR() returns binary SDR of a bitmap
  - bitmapFromSDR() returns the bitmap rep of an SDR
  - pprintHeader() prints a header describing the encoding to the terminal
  - pprint() prints an encoding to the terminal
  - decodedToStr() returns pretty print string of decoded SDR

  Methods/properties that must be implemented by subclasses:
  - encode() returns a numpy array encoding the input
  - decode() returns a list of strings representing a decoded SDR
  - getWidth() returns the output width, in bits
  - getDescription() returns a dict describing the encoded output
  """

  def __init__(self, n=DEFAULT_N, w=DEFAULT_W, unionSparsity=0.20):
    """The SDR dimensions are standard for Cortical.io fingerprints."""
    self.n = n
    self.w = w
    self.unionSparsity = unionSparsity
    self.targetSparsity = float(self.w) / self.n


  def encode(self, inputText):
    """
    Encodes inputText and puts the encoded value into the numpy output array,
    which is a 1-D array of length returned by getWidth().
    Note: The numpy output array is reused, so clear it before updating it.

    @param inputData      (str)     Data to encode. This should be validated by
                                    the encoder subclass.
    @param output         (numpy)   1-D array of same length returned by
                                    getWidth().
    """
    raise NotImplementedError


  def encodeIntoArray(self, inputText, output):
    """
    Encodes inputData and puts the encoded value into the numpy output array,
    which is a 1-D array of length returned by getWidth().

    Note: The numpy output array is reused, so clear it before updating it.

    @param inputData Data to encode. This should be validated by the encoder.
    @param output numpy 1-D array of same length returned by getWidth()
    """
    raise NotImplementedError


  def decode(self, encoded):
    """
    Decodes the SDR encoded. See subclass implementation for details; the
    decoding approaches and return objects vary depending on the encoder.

    To pretty print the return value from this method, use decodedToStr().

    @param encoded        (numpy)     Encoded 1-d array (an SDR).
    """
    raise NotImplementedError


  def getWidth(self):
    """
    Get an encoding's output width in bits. See subclass implementation for
    details.
    """
    raise NotImplementedError()


  def getDescription(self):
    """
    Returns a tuple, each containing (name, offset).
    The name is a string description of each sub-field, and offset is the bit
    offset of the sub-field for that encoder; should be 0.
    """
    raise NotImplementedError()


  def bitmapToSDR(self, bitmap):
    """Convert SDR encoding from bitmap to binary numpy array."""
    sdr = numpy.zeros(self.n)
    sdr[bitmap] = 1
    return sdr


  def bitmapFromSDR(self, sdr):
    """Convert SDR encoding from binary numpy array to bitmap."""
    return numpy.array([i for i in range(len(sdr)) if sdr[i]==1])


  def encodeRandomly(self, text, w, n):
    """Return a random bitmap representation of the text."""
    random.seed(text)
    return numpy.sort(random.sample(xrange(n), w))


  def compare(self, bitmap1, bitmap2):
    """
    Compare bitmaps, returning a dict of similarity measures.

    @param bitmap1     (list)        Indices of ON bits.
    @param bitmap2     (list)        Indices of ON bits.
    @return distances  (dict)        Key-values of distance metrics and values.

    Example return dict:
      {
        "cosineSimilarity": 0.6666666666666666,
        "euclideanDistance": 0.3333333333333333,
        "jaccardDistance": 0.5,
        "overlappingAll": 6,
        "overlappingLeftRight": 0.6666666666666666,
        "overlappingRightLeft": 0.6666666666666666,
        "sizeLeft": 9,
        "sizeRight": 9
      }
    """
    if not len(bitmap1) > 0 or not len(bitmap2) > 0:
      raise ValueError("Bitmaps must have ON bits to compare.")

    sdr1 = self.bitmapToSDR(bitmap1)
    sdr2 = self.bitmapToSDR(bitmap2)

    distances = {
      "sizeLeft": float(len(bitmap1)),
      "sizeRight": float(len(bitmap2)),
      "overlappingAll": float(len(numpy.intersect1d(bitmap1, bitmap2))),
      "euclideanDistance": numpy.linalg.norm(sdr1 - sdr2)
    }

    distances["overlappingLeftRight"] = (distances["overlappingAll"] /
                                         distances["sizeLeft"])
    distances["overlappingRightLeft"] = (distances["overlappingAll"] /
                                         distances["sizeRight"])
    distances["cosineSimilarity"] = (distances["overlappingAll"] /
        (math.sqrt(distances["sizeLeft"]) * math.sqrt(distances["sizeRight"])))
    distances["jaccardDistance"] = 1 - (distances["overlappingAll"] /
        len(numpy.union1d(bitmap1, bitmap2)))

    return distances


  def sparseUnion(self, counts):
    """
    Bits from the input patterns are unionized and then sparsified.

    @param counts     (Counter)   A count of the ON bits for the union bitmap.

    @return           (list)      A sparsified union bitmap.
    """
    max_sparsity = int(self.unionSparsity * self.n)
    w = min(len(counts), max_sparsity)
    return [c[0] for c in counts.most_common(w)]  # TODO: how does this break ties?


  @staticmethod
  def scaleEncoding(encoding, scaleFactor):
    """
    Scale down the size of the encoding by a factor between 0 and 1.

    @param encoding     (array)   Bitmap encoding.
    @param scaleFactor  (float)   Factor between 0 and 1 to scale down the SDR's
                                  size (n) by.
    @return             (list)    Scaled down bitmap of the encoding.
    """
    scaledBitmap = [int(i * float(scaleFactor)) for i in encoding]
    return sorted(set(scaledBitmap), key=lambda x: scaledBitmap.index(x))


  def pprintHeader(self, prefix=""):
    """
    Pretty-print a header that labels the sub-fields of the encoded output.
    This can be used in conjuction with pprint().
    @param prefix printed before the header if specified
    """
    print prefix,
    description = self.getDescription() + [("end", self.getWidth())]
    for i in xrange(len(description) - 1):
      name = description[i][0]
      width = description[i+1][1] - description[i][1]
      formatStr = "%%-%ds |" % width
      if len(name) > width:
        pname = name[0:width]
      else:
        pname = name
      print formatStr % pname,
    print
    print prefix, "-" * (self.getWidth() + (len(description) - 1)*3 - 1)


  def pprint(self, output, prefix=""):
    """
    Pretty-print the encoded output using ascii art.
    @param output to print
    @param prefix printed before the header if specified
    """
    print prefix,
    description = self.getDescription() + [("end", self.getWidth())]
    for i in xrange(len(description) - 1):
      offset = description[i][1]
      nextoffset = description[i+1][1]
      print "%s |" % bitsToString(output[offset:nextoffset]),
    print


  def decodedToStr(self, decodeResults):
    """
    Return a pretty print string representing the return value from decode().
    """

    (fieldsDict, fieldsOrder) = decodeResults

    desc = ''
    for fieldName in fieldsOrder:
      (ranges, rangesStr) = fieldsDict[fieldName]
      if len(desc) > 0:
        desc += ", %s:" % (fieldName)
      else:
        desc += "%s:" % (fieldName)

      desc += "[%s]" % (rangesStr)

    return desc
