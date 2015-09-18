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
import os
from collections import Counter

from cortipy.cortical_client import CorticalClient
from cortipy.exceptions import UnsuccessfulEncodingError
from htmresearch.encoders import EncoderTypes
from htmresearch.encoders.language_encoder import LanguageEncoder
from support.text_preprocess import TextPreprocess


DEFAULT_RETINA = "en_synonymous"



class CioEncoder(LanguageEncoder):
  """
  A language encoder using the Cortical.io API.

  The encoder queries the Cortical.io REST API via the cortipy module, which
  returns data in the form of "fingerprints". These representations are
  converted to binary SDR arrays with this Cio encoder.
  """

  def __init__(self, w=128, h=128, retina=DEFAULT_RETINA, cacheDir=None,
               verbosity=0, fingerprintType=EncoderTypes.document,
               unionSparsity=20.0):
    """
    @param w               (int)      Width dimension of the SDR topology.
    @param h               (int)      Height dimension of the SDR topology.
    @param cacheDir        (str)      Where to cache results of API queries.
    @param verbosity       (int)      Amount of info printed out, 0, 1, or 2.
    @param fingerprintType (Enum)     Specify word- or document-level encoding.
    """
    if "CORTICAL_API_KEY" not in os.environ:
      print ("Missing CORTICAL_API_KEY environment variable. If you have a "
        "key, set it with $ export CORTICAL_API_KEY=api_key\n"
        "You can retrieve a key by registering for the REST API at "
        "http://www.cortical.io/resources_apikey.html")
      raise OSError("Missing API key.")

    super(CioEncoder, self).__init__(unionSparsity = unionSparsity)

    if cacheDir is None:
      root = os.path.dirname(os.path.realpath(__file__))
      cacheDir = os.path.join(root, "CioCache")

    self.apiKey = os.environ["CORTICAL_API_KEY"]
    self.client = CorticalClient(self.apiKey, retina=retina, cacheDir=cacheDir)
    self.w = w
    self.h = h
    self.n = w*h
    self.verbosity = verbosity
    self.fingerprintType = fingerprintType
    self.description = ("Cio Encoder", 0)


  def encode(self, text):
    """
    Encodes the input text w/ a cortipy client. The client returns a
    dictionary of "fingerprint" info, including the SDR bitmap.

    NOTE: returning this fingerprint dict differs from the base class spec.

    @param  text    (str)             A non-tokenized sample of text.
    @return         (dict)            Result from the cortipy client. The bitmap
                                      encoding is at
                                      encoding["fingerprint"]["positions"].
    """
    if not text:
      return None
    try:
      if self.fingerprintType == EncoderTypes.document:
        encoding = self.client.getTextBitmap(text)
      elif self.fingerprintType == EncoderTypes.word:
        encoding = self.getUnionEncoding(text)
    except UnsuccessfulEncodingError:
      if self.verbosity > 0:
        print ("\tThe client returned no encoding for the text \'{0}\', so "
               "we'll use the encoding of the token that is least frequent in "
               "the corpus.".format(text))
      encoding = self._subEncoding(text)

    return encoding


  def getUnionEncoding(self, text):
    """
    Encode each token of the input text, take the union, and then sparsify.

    @param  text    (str)             A non-tokenized sample of text.
    @return         (dict)            The bitmap encoding is at
                                      encoding["fingerprint"]["positions"].
    """
    tokens = TextPreprocess().tokenize(text)

    # Count the ON bits represented in the encoded tokens.
    counts = Counter()
    for t in tokens:
      bitmap = self.client.getBitmap(t)["fingerprint"]["positions"]
      counts.update(bitmap)

    positions = self.sparseUnion(counts)

    # Populate encoding
    encoding = {
        "text": text,
        "sparsity": len(positions) * 100 / float(self.n),
        "df": 0.0,
        "height": self.h,
        "width": self.w,
        "score": 0.0,
        "fingerprint": {
            "positions":sorted(positions)
            },
        "pos_types": []
        }

    return encoding


  def encodeIntoArray(self, inputText, output):
    """
    See method description in language_encoder.py. It is expected the inputText
    is a single word/token (str).

    NOTE: nupic Encoder class method encodes output in place as sparse array
    (commented out below), but this method returns a bitmap.
    """
    if not isinstance(inputText, str):
      raise TypeError("Expected a string input but got input of type {}."
                      .format(type(inputText)))

    # Encode with term endpoint of Cio API
    try:
      encoding = self.client.getBitmap(inputText)
    except UnsuccessfulEncodingError:
      if self.verbosity > 0:
        print ("\tThe client returned no encoding for the text \'{0}\', so "
               "we'll use the encoding of the token that is least frequent in "
               "the corpus.".format(inputText))
      encoding = self._subEncoding(inputText)

    # output = sparsify(encoding["fingerprint"]["positions"])
    return encoding


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
    return [((term["term"], term["score"])) for term in terms]


  def _subEncoding(self, text, method="keyword"):
    """
    @param text             (str)             A non-tokenized sample of text.
    @return encoding        (dict)            Fingerprint from cortipy client.
                                              An empty dictionary of the text
                                              could not be encoded.
    """
    tokens = list(itertools.chain.from_iterable(
        [t.split(',') for t in self.client.tokenize(text)]))
    try:
      if method == "df":
        encoding = min([self.client.getBitmap(t) for t in tokens],
                       key=lambda x: x["df"])
      elif method == "keyword":
        encoding = self.getUnionEncoding(text)
      else:
        raise ValueError("method must be either \'df\' or \'keyword\'")
    except UnsuccessfulEncodingError:
      if self.verbosity > 0:
        print ("\tThe client returned no substitute encoding for the text "
               "\'{0}\', so we encode with None.".format(text))
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


  def getDescription(self):
    return self.description
