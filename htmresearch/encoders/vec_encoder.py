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

import numpy

from collections import Counter, OrderedDict
from htmresearch.encoders.language_encoder import LanguageEncoder
from htmresearch.support.text_preprocess import TextPreprocess
from nupic.encoders.base import Encoder
from nupic.encoders.scalar import ScalarEncoder



class VecEncoder(LanguageEncoder):
  """
  A language encoder for word embeddings; dense distributed vectors
  (word embeddings) are encoded into SDRs.
  """

  def __init__(self,
               embeddingsPath,
               vecSize=200,
               n=16384,
               unionSparsity=20.0,
               verbosity=0):
    """
    @param vecSize    (int)     Length of word embeddings.
    @param N          (int)     Length of vector encoding.
    """
    nMicro=n/vecSize  # 16384/200=81
    wMicro=5  # 5/81=0.0617 --> 0.0610 sparsity overall
    # wMicro=3  # 3/81=0.037 --> 0.0366
    super(VecEncoder, self).__init__(n=n, unionSparsity = unionSparsity)

    self.vecSize = vecSize
    self.wMacro = wMicro * vecSize
    self.nMacro = n
    self.unionSparsity = unionSparsity

    self.description = ("Word Vector Encoder", 0)
    self.verbosity=verbosity

    self.entries, minVal, maxVal = self.getEmbeddings(embeddingsPath)

    self.microEncoder = ScalarEncoder(wMicro, minVal, maxVal, n=nMicro, forced=True)
    self.macroEncoder = VectorEncoder(self.vecSize, self.microEncoder, name="wordVec")


  def getEmbeddings(self, embeddingsPath):
    """
    Expects GloVe format. Returns entries dict, min value, and max value.
    """
    with open(embeddingsPath, "rb") as f:
      print "Reading in vectors..."

      entries = OrderedDict()
      maxVal = 0.0
      minVal = 0.0
      for i, l in enumerate(f):
        line = l.split(' ')
        temp = line[1:]
        temp[-1] = temp[-1][:-2]

        try:
          entries[line[0]] = [float(value) for value in temp]
          maxThis = max(entries[line[0]])
          if maxThis > maxVal:
            maxVal = maxThis
          minThis = min(entries[line[0]])
          if minThis < minVal:
            minVal = minThis
        except Exception as e:
          print "\t", e, line[0]

    print "\tmax value = ", maxVal
    print "\tmin value = ", minVal

    return entries, minVal, maxVal


  def encode(self, text):
    """
    Encode each word of the input text according to its word embedding vector,
    take the union, and then sparsify.

    @param  text    (str)             A tokenized sample of text.
    @return         (dict)            The text, its bitmap, and sparsity.
    """
    if not text: return

    counts = Counter()
    for word in text:
      sparseVec = self.encodeEmbedding(word)
      if sparseVec is not None:
        bitmap = numpy.where(sparseVec==1)[0]
      else:
        bitmap = self.encodeRandomly(word, self.wMacro, self.nMacro)
      counts.update(bitmap)

    positions = self.sparseUnion(counts)

    encoding = {
      "text": text,
      "sparsity": len(positions) / float(self.nMacro),
      "bitmap": sorted(positions),
    }

    return encoding


  def encodeEmbedding(self, word):
    """Encode the word's vector representation."""
    wordVec = self.entries.get(word)
    if wordVec is None: return
    encoding = numpy.zeros(self.nMacro)
    self.macroEncoder.encodeIntoArray([float(v) for v in wordVec], encoding)
    return encoding


  def writeOutEncodings(self, encodings, outPath=None):
    """Write to JSON."""
    if outPath is None:
      outPath = os.path.join(os.path.dirname(os.path.realpath(__file__)), "wordVecSDR.json")

    print "Writing encodings to {}...".format(outPath)

    with open(outPath, "w") as f:
      json.dump(encodings, f)



# VectorEncoder class as previously implemented in NuPIC:
class VectorEncoder(Encoder):
  """represents an array/vector of values of the same type (scalars, or date, ..);"""

  def __init__(self, length, encoder, name='vector', typeCastFn=None):
    """param length: size of the vector, number of elements
       param encoder: instance of encoder used for coding of the elements
       param typeCastFn: function to convert decoded output (as string) back to original values
       NOTE: this constructor cannot be used in description.py, as it depands passing of an object!
    """

    if not (isinstance(length, int) and length > 0):
      raise Exception("Length must be int > 0")
    if not isinstance(encoder, Encoder):
      raise Exception("Must provide an encoder")
    if typeCastFn is not None and not isinstance(typeCastFn, type):
      raise Exception("if typeCastFn is provided, it must be a function")

    self._len = length
    self._enc = encoder
    self._w = encoder.getWidth()
    self._name = name
    self._typeCastFn = typeCastFn


  def encodeIntoArray(self, input, output):
    if not isinstance(input, list) and len(input)==self._len:
      raise Exception("input must be list if size %d" % self._len)
    for e in xrange(self._len):
      tmp = self._enc.encode(input[e])
      output[e*self._w:(e+1)*self._w] = tmp


  def decode(self, encoded, parentFieldName=''):
    ret = []
    w = self._w
    for i in xrange(self._len):
      tmp = self._enc.decode(encoded[i*w:(i+1)*w])[0].values()[0][1] # dict.values().first_element.scalar_value
      if self._typeCastFn is not None:
        tmp = self._typeCastFn(tmp)
      ret.append(tmp)

    # Return result as EncoderResult
    if parentFieldName != '':
      fieldName = "%s.%s" % (parentFieldName, self._name)
    else:
      fieldName = self._name
    ranges = ret
    desc = ret
    return ({fieldName: (ranges, desc)}, [fieldName])

  def getData(self, decoded):
    """get the data part (vector) from the decode() output format;
       use when you want to work with the data manually"""
    fieldname = decoded[1][0]
    return decoded[0][fieldname][0]

  def getWidth(self):
    return self._len * self._enc.getWidth()
