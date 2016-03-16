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
import itertools
import numpy
import os

import random

from cortipy.cortical_client import CorticalClient, RETINA_SIZES
from cortipy.exceptions import UnsuccessfulEncodingError
from htmresearch.encoders import EncoderTypes
from htmresearch.encoders.language_encoder import LanguageEncoder
from htmresearch.support.text_preprocess import TextPreprocess

from nupic.bindings.math import SparseMatrix


DEFAULT_RETINA = "en_synonymous"



class CioEncoder(LanguageEncoder):
  """
  A language encoder using the Cortical.io API.

  The encoder queries the Cortical.io REST API via the cortipy module, which
  returns data in the form of "fingerprints". These representations are
  converted to binary SDR arrays with this Cio encoder.
  """

  def __init__(self, retina=DEFAULT_RETINA, retinaScaling=1.0, cacheDir=None,
               verbosity=0, fingerprintType=EncoderTypes.document,
               unionSparsity=0.20, apiKey=None,
               maxSparsity=0.50):
    """
    @param retina          (str)      Cortical.io retina, either "en_synonymous"
                                      or "en_associative".
    @param retinaScaling   (float)    Scale each dimension of the SDR bitmap
                                      by this factor.
    @param cacheDir        (str)      Where to cache results of API queries.
    @param verbosity       (int)      Amount of info printed out, 0, 1, or 2.
    @param fingerprintType (Enum)     Specify word- or document-level encoding.
    @param unionSparsity   (float)    Any union'ing done in this encoder will
                                      stop once this sparsity is reached.
    @param maxSparsity     (float)    The maximum sparsity of the returned
                                      bitmap. If the percentage of bits in the
                                      encoding is > maxSparsity, it will be
                                      randomly subsampled.

    TODO: replace enum with a simple string
    """
    if apiKey is None and "CORTICAL_API_KEY" not in os.environ:
      print ("Missing CORTICAL_API_KEY environment variable. If you have a "
        "key, set it with $ export CORTICAL_API_KEY=api_key\n"
        "You can retrieve a key by registering for the REST API at "
        "http://www.cortical.io/resources_apikey.html")
      raise OSError("Missing API key.")

    super(CioEncoder, self).__init__(unionSparsity=unionSparsity)
    # import pdb; pdb.set_trace()
    if cacheDir is None:
      root = os.path.dirname(os.path.realpath(__file__))
      cacheDir = os.path.join(root, "CioCache")

    self.apiKey = apiKey if apiKey else os.environ["CORTICAL_API_KEY"]
    self.client = CorticalClient(self.apiKey, retina=retina, cacheDir=cacheDir)

    self.cacheDir = cacheDir
    self.client = CorticalClient(self.apiKey,
                                 retina=retina,
                                 cacheDir=self.cacheDir)

    self._setDimensions(retina, retinaScaling)

    self.fingerprintType = fingerprintType
    self.description = ("Cio Encoder", 0)

    self.verbosity = verbosity
    self.maxSparsity = maxSparsity


  @property
  def cacheDir(self):
     """ cacheDir property with unique property that if the user originally
     passed in an explicit cacheDir to CioEncoder constructor, then that value
     is used.  If not, the default value is calculated.  This is done in
     conjunction with _setCacheDir() which only sets self._cacheDir if a value
     is provided in the setter.
     """
     if hasattr(self, "_cacheDir"):
       return self._cacheDir

     # If user never explicitly sets cacheDir, return calculated cacheDir
     root = os.path.dirname(os.path.realpath(__file__))
     return os.path.join(root, "CioCache")


  @cacheDir.setter
  def cacheDir(self, value):
    if value:
      # Only set cacheDir if value explicitly provided
      self._cacheDir = value


  def __setstate__(self, state):
    """ Called when CioEncoder is unpickled per pickle protocol.  This includes
    a mechanism to gracefully re-init the CorticalClient instance with a
    calculated cacheDir, in the event that the previously pickled
    CorticalClient instance includes a cacheDir that does not exist, which is
    likely the case when a model is trained on one machine for reuse elsewhere.
    """
    if "_cacheDir" not in state:
      state["client"] = CorticalClient(state["apiKey"],
                                       retina=state["client"].retina,
                                       cacheDir=self.cacheDir)

    self.__dict__ = state


  def _setDimensions(self, retina, scalingFactor):
    if scalingFactor <= 0 or scalingFactor > 1:
      raise ValueError("Retina can only be scaled by values between 0 and 1.")

    retinaDim = RETINA_SIZES[retina]["width"]

    self.width = int(retinaDim * scalingFactor)
    self.height = int(retinaDim * scalingFactor)
    self.retinaScaling = float(self.width)/retinaDim
    self.n = self.width * self.height


  def encode(self, text):
    """
    Encodes the input text w/ a cortipy client. The client returns a
    dictionary of "fingerprint" info, including the SDR bitmap as a numpy.array.

    NOTE: returning this fingerprint dict differs from the base class spec.

    @param  text    (str)             A non-tokenized sample of text.
    @return         (dict)            Result from the cortipy client. The bitmap
                                      encoding is at
                                      encoding["fingerprint"]["positions"].
    """
    if not isinstance(text, str) and not isinstance(text, unicode):
      raise TypeError("Expected a string input but got input of type {}."
                      .format(type(text)))

    try:
      if self.fingerprintType == EncoderTypes.document:
        encoding = self.client.getTextBitmap(text)

      elif self.fingerprintType == EncoderTypes.word:
        encoding = self.getUnionEncoding(text)

      else:
        encoding = self.client.getBitmap(text)

    except UnsuccessfulEncodingError:
      if self.verbosity > 0:
        print ("\tThe client returned no encoding for the text \'{0}\', so "
               "we'll use the encoding of the token that is least frequent in "
               "the corpus.".format(text))
      encoding = self._subEncoding(text)

    return self.finishEncoding(encoding)


  def getUnionEncodingFromTokens(self, tokens):
    """
    Create a single, sparsified bitmap from a union of bitmaps for given tokens

    @param  tokens  (sequence) A sequence of tokens
    @return         (sequence) Bitmap
    """
    # Accumulate counts by inplace-adding sparse matrices
    counts = SparseMatrix()
    counts.resize(1, self.width*self.height)

    # Pre-allocate buffer sparse matrix
    sparseBitmap = SparseMatrix()
    sparseBitmap.resize(1, self.width*self.height)

    for t in tokens:
      bitmap = self._getWordBitmap(t)
      sparseBitmap.setRowFromSparse(0, bitmap, [1]*len(bitmap))
      counts += sparseBitmap

    numNonZeros = counts.nNonZerosOnRow(0)

    # Add some jitter to aid in tie-breaking during sort
    jitterValues = tuple((random.random()/100)+0.1
                         for _ in xrange(numNonZeros))
    sparseBitmap.setRowFromSparse(0, counts.nonZeroCols(), jitterValues)
    counts += sparseBitmap

    maxSparsity = int(self.unionSparsity * self.n)
    w = min(numNonZeros, maxSparsity)

    # Return top w most popular positions
    return  tuple(x[1] for x in counts.getNonZerosSorted())[:w]


  def getUnionEncoding(self, text):
    """
    Encode each token of the input text, take the union, and then sparsify.

    @param  text    (str)             A non-tokenized sample of text.
    @return         (dict)            The bitmap encoding is at
                                      encoding["fingerprint"]["positions"].
    """
    tokens = TextPreprocess().tokenize(text)

    positions = self.getUnionEncodingFromTokens(tokens)

    # Populate encoding
    encoding = {
        "text": text,
        "sparsity": len(positions) / float(self.n),
        "df": 0.0,
        "height": self.height,
        "width": self.width,
        "score": 0.0,
        "fingerprint": {
            "positions":sorted(positions)
            },
        "pos_types": []
        }

    return encoding


  def getWindowEncoding(self, tokens, minSparsity=0.0):
    """
    The encodings simulate a "sliding window", where the encoding representation
    of a given token is a union of its bitmap with the immediately previous
    tokens' bitmaps, up to the maximum sparsity. The returned list only includes
    those windows with sparsities larger than the minimum.

    @param tokens           (list)  Tokenized string.
    @param minSparsity      (float) Only window encodings denser than this value
                                    will be included.
    @return windowBitmaps   (list)  Dict for each token, with entries for the
                                    token string, sparsity float, and bitmap
                                    numpy array.
    """
    if self.fingerprintType != EncoderTypes.word:
      print ("Although the encoder type is not set for words, the window "
        "encodings use word-level fingerprints.")

    bitmaps = tuple(numpy.array(self._getWordBitmap(t)) for t in tokens)

    windowBitmaps = []
    for tokenIndex, windowBitmap in enumerate(bitmaps):
      # Each index in the tokens list is the end of a possible window.
      for i in reversed(xrange(tokenIndex)):
        # From the current token, increase the window by successively adding the
        # previous tokens.
        windowSparsity = len(windowBitmap) / float(self.n)
        nextSparsity = len(bitmaps[i]) / float(self.n)
        if windowSparsity + nextSparsity > self.unionSparsity:
          # stopping criterion reached -- window is full
          break
        else:
          # add bitmap to the current window bitmap
          windowBitmap = numpy.union1d(windowBitmap, bitmaps[i])

      sparsity = len(windowBitmap) / float(self.n)
      if sparsity > minSparsity:
        # only include windows of sufficient density
        windowBitmaps.append(
          {"text": tokens[i:tokenIndex+1],
           "sparsity": sparsity,
           "bitmap": numpy.array(windowBitmap)})

    return windowBitmaps


  def finishEncoding(self, encoding):
    """
    Scale the fingerprint of the encoding dict (if specified) and fill the
    width, height, and sparsity fields.

    @param encoding       (dict)      Dict as returned by the Cio client.
    @return encoding      (dict)      Same format as the input dict, with the
                                      dimensions and sparsity fields populated.
    """
    if self.retinaScaling != 1:
      encoding["fingerprint"]["positions"] = self.scaleEncoding(
        encoding["fingerprint"]["positions"], self.retinaScaling**2)
      encoding["width"] = self.width
      encoding["height"] = self.height

    encoding["fingerprint"]["positions"] = numpy.array(
      encoding["fingerprint"]["positions"])

    encoding["sparsity"] = len(encoding["fingerprint"]["positions"]) / float(
      (encoding["width"] * encoding["height"]))

    # Reduce sparsity if needed
    if encoding["sparsity"] > self.maxSparsity:
      self.reduceSparsity(encoding, self.maxSparsity)

    return encoding


  def _getWordBitmap(self, term):
    """
    Return a bitmap for the word. If the Cortical.io API can't encode, cortipy
    will use a random encoding for the word.
    """
    return self.client.getBitmap(term)["fingerprint"]["positions"]


  def encodeIntoArray(self, inputText, output):
    """
    Encodes inputText and puts the encoded value into the numpy output array,
    which is a 1-D array of length returned by getWidth().
    """
    encoding = self.encode(inputText)
    output[:] = 0
    if encoding["fingerprint"]["positions"].size > 0:
      output[encoding["fingerprint"]["positions"]] = 1


  def decode(self, encoding, numTerms=10):
    """
    Converts an SDR back into the most likely word or words.

    By default, the most likely term will be returned. If numTerms is
    specified, then the Cortical.io API will attempt to return that many;
    otherwise the standard is 10. The return value will be a sequence of
    (term, weight) tuples, where higher weights imply the corresponding term
    better matches the encoding.

    @param  encoding        (list)            Bitmap encoding.
    @param  numTerms        (int)             The max number of terms to return.
    @return                 (list)            List of dictionaries, where keys
                                              are terms and likelihood scores.
    """
    terms = self.client.bitmapToTerms(encoding, numTerms=numTerms)
    # Convert cortipy response to list of tuples (term, weight)
    return [(term["term"], term["score"]) for term in terms]


  def _subEncoding(self, text, method="keyword"):
    """
    @param text             (str)             A non-tokenized sample of text.
    @return encoding        (dict)            Fingerprint from cortipy client.
                                              An empty dictionary of the text
                                              could not be encoded.
    """
    try:
      if method == "df":
        tokens = list(itertools.chain.from_iterable(
          [t.split(",") for t in self.client.tokenize(text)]))
        encoding = min(
          [self.client.getBitmap(t) for t in tokens], key=lambda x: x["df"])
      elif method == "keyword":
        encoding = self.getUnionEncoding(text)
      else:
        raise ValueError("method must be either 'df' or 'keyword'")
    except UnsuccessfulEncodingError:
      if self.verbosity > 0:
        print ("\tThe client returned no substitute encoding for the text "
               "'{}', so we encode with None.".format(text))
      encoding = None

    return encoding


  def compare(self, bitmap1, bitmap2):
    """
    Compare encodings, returning the distances between the SDRs. Input bitmaps
    must be list objects (need to be serializable).

    Example return dict:
      {
        "cosineSimilarity": 0.6666666666666666,
        "euclideanDistance": 0.3333333333333333,
        "jaccardDistance": 0.5,
        "overlappingAll": 6,
        "overlappingLeftRight": 0.6666666666666666,
        "overlappingRightLeft": 0.6666666666666666,
        "sizeLeft": 9,
        "sizeRight": 9,
        "weightedScoring": 0.4436476984102028
      }
    """
    if not isinstance(bitmap1 and bitmap2, list):
      raise TypeError("Comparison bitmaps must be lists.")

    return self.client.compare(bitmap1, bitmap2)


  def createCategory(self, label, positives, negatives=None):
    """
    Create a classification category (bitmap) via the Cio claassify endpoint.

    @param label      (str)     Name of category.
    @param positives  (list)    Bitmap(s) of samples to define.
    @param negatives  (list)    Not required to make category.

    @return           (dict)    Key-values for "positions" (list bitmap encoding
                                of the category and "categoryName" (str).
    """
    if negatives is None:
      negatives = []
    if not isinstance(positives and negatives, list):
      raise TypeError("Input bitmaps must be lists.")

    return self.client.createClassification(label, positives, negatives)


  def getWidth(self):
    return self.n


  def getDimensions(self):
    return (self.width, self.height)


  def getDescription(self):
    return self.description


  def densifyPattern(self, bitmap):
    """Return a numpy array of 0s and 1s to represent the given bitmap."""
    sparsePattern = numpy.zeros(self.n)
    for i in bitmap:
      sparsePattern[i] = 1.0
    return sparsePattern


  def reduceSparsity(self, encoding, maxSparsity):
    """Reduce the sparsity of the encoding down to maxSparsity"""

    desiredBits = maxSparsity*encoding["width"]*encoding["height"]
    bitmap = encoding["fingerprint"]["positions"]

    # Choose a random subsampling of the bits but seed the random number
    # generator so we get consistent bitmaps
    numpy.random.seed(bitmap.sum())
    encoding["fingerprint"]["positions"] = (
      numpy.random.permutation(bitmap)[0:desiredBits] )

    encoding["sparsity"] = len(encoding["fingerprint"]["positions"]) / float(
      (encoding["width"] * encoding["height"]))
