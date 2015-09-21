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

import collections
import gensim
import numpy
import operator

from htmresearch.encoders.language_encoder import LanguageEncoder


TARGET_SPARSITY = 1.0
exclusions = ('!', '.', ':', ',', '"', '\'', '\n', '?')



class LSAEncoder(LanguageEncoder):
  """
  A language encoder using LSA model.

  The associated script must be used to generate the tf-idf and LSA models that
  are used by the encoder. The encoder takes arbitrary text, converts it to the
  topic space via the models, and then creates an SDR. The SDR has a bit for
  each topic. The top `w` topics are set to 1.
  """

  def __init__(self):
    self.dictionary = gensim.corpora.Dictionary.load_from_text(
        "wiki/wiki_en_wordids.txt")
    self.tfidf = gensim.models.TfidfModel.load(tfidfModelPath)
    self.lsa = gensim.models.lsimodel.LsiModel.load(languageModelPath)

    self.n = self.lsa.num_topics
    if width:
      self.w = w
    else:
      self.w = int(float(self.n) * 0.05)

    self.description = ("LSA Encoder", 0)


  def _tokenize(self, text):
    """Tokenize the text string into a list of strings."""
    text = "".join([c for c in text if c not in exclusions])
    return text.split(" ")


  def encode(self, text):
    """
    Encodes the input text into an SDR.

    @param  text    (str, list)       If the input is type str, the encoder
                                      assumes it has not yet been tokenized. A
                                      list input will skip the tokenization
                                      step.
    @return         (list)            SDR.

    TODO: test tokenization logic for str and list inputs
    """
    if isinstance(text, str):
      text = self._tokenize(text)

    bow = self.dictionary.doc2bow(text.lower().split())
    tfidf = self.tfidf[bow]
    weights = self.lsa[tfidf]
    topWeights = sorted(weights, key=operator.itemgetter(1))[-self.w:]
    activeIndices = [pair[0] for pair in topWeights]
    encoded = numpy.zeros([self.n], dtype=numpy.bool)
    encoded[activeIndices] = 1
    return encoded


  def encodeIntoArray(self, inputText, output):
    """See method description in language_encoder.py."""
    if not isinstance(inputText, str):
      raise TypeError("Expected a string input but got input of type {}."
                      .format(type(inputText)))

    ## TODO
    pass


  def decode(self, encoding, numTerms=None):
    """Converts an SDR back into the most likely word or words.

    By default, the most likely term will be returned. If numTerms is
    specified then it determines how many terms will be returned and the
    return value will be a sequence of (term, weight) tuples where the
    higher the weight, the more the term matches the encoding.
    """
    potentialTerms = collections.defaultdict(float)
    for topicNum in xrange(self.lsa.num_topics):
      if encoding[topicNum] == 1:
        for weight, term in self.lsa.show_topic(topicNum):
          potentialTerms[term] += weight

    # Temporary variable for how many terms to get
    n = numTerms if numTerms is not None else 1
    topTerms = sorted(potentialTerms.items(),
                      key=operator.itemgetter(1),
                      reverse=True)[:n]
    # If numTerms is not specified, return just the most likely term, otherwise
    # return the top numTerms terms with weights.
    if numTerms is None:
      return topTerms[0][0]
    else:
      return topTerms


  def getWidth(self):
    return self.n


  def getDescription(self):
    return self.description
