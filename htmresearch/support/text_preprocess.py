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
"""
This file contains text pre-processing functions for NLP experiments.
"""

import os
import pandas
import re
import string

from collections import Counter
from functools import partial



class TextPreprocess(object):
  """Class for text pre-processing"""

  alphabet = string.ascii_lowercase

  def __init__(self,
               corpusTxt="compilation.txt",
               abbrCSV="abbreviations.csv",
               contrCSV="contractions.csv"):
    """
    @param corpusTxt      (str)       A compilation of most frequent words. The
        default file 'compilation.txt' is the most frequent words sourced from
        British National Corpus, Wiktionary, and the Project Guttenberg books.

    @param abbrCSV        (str)       A compilation of domain specific
        abbreviations. The file is a csv with the header "Abbr,Expansion". The
        default file 'abbreviations.csv' contains basic abbreviations.

    @param contrCSV       (str)       A compilation of common contractions. The
        file is a csv with the header "Contr,Expansion". The default file
        'contractions.csv' contains a short list of common contractions.
    """
    self.abbrCSV = abbrCSV
    self.contrCSV = contrCSV
    self.corpusTxt = corpusTxt

    self.abbrs = None
    self.abbrRegex = None
    self.bagOfWords = None
    self.contrs = None
    self.contrRegex = None
    self.corpus = None


  def _setupCorpus(self, corpusSource):
    """Create member vars for English language corpus and bag of words."""
    try:
      if os.path.isfile(corpusSource):
        # absolute path
        corpusPath = corpusSource
      else:
        # relative path
        corpusPath = os.path.abspath(os.path.join(
          os.path.dirname(__file__), "..", "..", "projects/nlp/data/etc",
          corpusSource))
      self.corpus = file(corpusPath).read()

    except IOError as e:
      print "Error reading in the text corpus file '{}'".format(filename)
      raise e

    self.bagOfWords = Counter(self.tokenize(self.corpus))


  def _setupAbbrs(self, abbrsSource):
    """
    Read in abbreviations, and combine all into one regex that will only match
    exact matches and not words containing them.
    E.g. if "WFH" is an abbreviation, it will match "WFH", but not "XWFH".
    """
    self.abbrs = self.readExpansionFile(abbrsSource, ["", "s", "'s"])
    self.abbrRegex = re.compile(r"\b%s\b" % r"\b|\b".join(self.abbrs.keys()))


  def _setupContr(self, contrSource):
    """
    Read in contractions, and combine all into one regex that will match any
    word ending in the contraction.
    E.g. if "'ll" is a contraction, it will match "he'll".
    """
    self.contrs = self.readExpansionFile(contrSource)
    self.contrRegex = re.compile(r"%s\b" % r"\b|".join(self.contrs.keys()))


  @staticmethod
  def readExpansionFile(filename, suffixes=None):
    """
    Read the csv file to get the original/expansion pairs and add suffixes if
    necessary.
    @param filename         (str)     Name of csv file to read. Expected format
                                      is original text in col 0, and expansion
                                      in col 1.
    @param suffixes         (list)    Strings that are added to the end of
                                      the original and expanded form if
                                      provided
    @return expansionPairs  (dict)    The keys are the original form with
                                      the suffixes added and the values are
                                      the expanded form with the suffixes
                                      added
    """
    if suffixes is None:
      suffixes = [""]

    expansionPairs = {}
    try:
      if os.path.isfile(filename):
        # absolute path
        path = filename
      else:
        # relative path
        path = os.path.abspath(os.path.join(
          os.path.dirname(__file__), "..", "..", "projects/nlp/data/etc",
          filename))
      dataFrame = pandas.read_csv(path)

      for i in xrange(dataFrame.shape[0]):
        original = dataFrame.iloc[i][0].lower()
        expansion = dataFrame.iloc[i][1].lower()

        for suffix in suffixes:
          originalSuffix = "{}{}".format(original, suffix)
          expSuffix = "{}{}".format(expansion, suffix)
          expansionPairs[originalSuffix] = expSuffix

    except IOError as e:
      print "Error reading in the expansion file '{}'".format(filename)
      raise e

    # Add an empty string if empty so the regex compiles
    if not expansionPairs:
      expansionPairs[""] = ""

    return expansionPairs


  def tokenizeAndFilter(self, text, **preprocessSpecs):
    """
    Tokenize and optionally filter the text, returning both the tokens and a
    mapping of the original text to the preprocessed text.

    @param text             (str)    Text to be tokenized and preprocessed.

    @return processedTokens (list)   String tokens.

    @return mapping         (list)   Map of tokens to their corresponding words
        in the original text. The original words are indexed via splitting on
        spaces. The list indices of mapping increment with consecutive tokens.

      E.g.
        text = "I wanna WFH!"
        ...
        processedTokens = ["want", "work", "from", "home"]
        mapping = [1, 2, 2, 2]  # e.g. the 3rd token points to the 2nd word
    """
    originalWords = text.split(" ")
    processedTokens = []
    mapping = []

    for i, word in enumerate(originalWords):
      preprocessed = self.tokenize(word)
      if preprocessSpecs:
        # preprocessing options are specified
        preprocessed = self._filterStuff(preprocessed, **preprocessSpecs)
      numTokens = len(preprocessed)
      mapping.extend([i] * numTokens)
      processedTokens.extend(preprocessed)

    return processedTokens, mapping


  @staticmethod
  def tokenize(text):
    """Tokenize, returning only lower-case letters and "$"."""
    if not isinstance(text, str) and not isinstance(text, unicode):
      raise TypeError(
          "{} is not an acceptable type for tokenization.".format(type(text)))

    return re.findall("[a-z$]+", text.lower())


  def _filterStuff(self,
                   tokenList,
                   ignoreCommon=None,
                   removeStrings=None,
                   correctSpell=False,
                   expandAbbr=False,
                   expandContr=False):
    """
    The order by which we filter the input text is by design:
      1. Expand abbreviations and contractions so we have the full set of
         possible words.
      2. Remove specified strings; these don't need to go through the rest of
         preprocessing.
      3. Correct spelling, which must precede the final filtering step.
      4. Remove the N most common words.

    @param tokenList      (list)  Lower-case alpha token(s), as returned by
                                  tokenize.
    @param ignoreCommon   (int)   The most frequent words up to this rank
                                  (in corpusTxt) will be removed.
    @param removeStrings  (list)  List of strings to delete from the text.
    @param correctSpell   (bool)  Correct the spelling; this utilizes a
                                  simplistic implementation and is not
                                  guaranteed to correct all misspellings.
    @param expandAbbr     (bool)  Expand abbreviations (in abbrCSV).
    @param expandContr    (bool)  Expand contractions (in contrCSV).

    @return tokens        (list)  Filtered tokens
    """
    # Any input tokens are expected to have been split on an apostrophe by the
    # tokenize() method.
    textString = string.join(tokenList, "'")

    if expandAbbr:
      if not self.abbrs:
        self._setupAbbrs(self.abbrCSV)
      getAbbrExpansion = partial(self.getExpansion, table=self.abbrs)
      textString = self.abbrRegex.sub(getAbbrExpansion, textString)

    if expandContr:
      if not self.contrs:
        self._setupContr(self.contrCSV)
      getContrExpansion = partial(self.getExpansion, table=self.contrs)
      textString = self.contrRegex.sub(getContrExpansion, textString)

    if removeStrings:
      for removal in removeStrings:
        textString = textString.replace(removal, "")

    tokens = self.tokenize(textString)

    if correctSpell:
      tokens = [self.correct(t) for t in tokens]

    if ignoreCommon:
      tokens = self.removeMostCommon(tokens, n=ignoreCommon)

    return tokens


  def removeMostCommon(self, tokenList, n=100):
    """
    Remove the n most common tokens as counted in the bag-of-words corpus.

    @param tokenList        (list)              List of token strings.
    @param n                (int)               Will filter out the n-most
                                                frequent terms.
    """
    if not self.bagOfWords:
      self._setupCorpus(self.corpusTxt)

    ignoreList = [word[0] for word in self.bagOfWords.most_common(n)]

    return [token for token in tokenList if token not in ignoreList]


  @staticmethod
  def getExpansion(match, table):
    """
    Gets the expanded version of the regular expression
    @param match            (_sre.SRE_Match)    Regex match to expand
    @param table            (dict)              Maps the string version of the
                                                regular expression to the
                                                string expansion
    @return                 (string)            Expansion
    """
    return table[match.string[match.start(): match.end()]]


  def correct(self, word):
    """
    Find the best spelling correction for this word. Prefer edit distance  of 0,
    then one, then two; otherwise default to the word itself.
    """
    if not self.bagOfWords:
      self._setupCorpus(self.corpusTxt)

    candidates = (self._known({word}) or
                  self._known(self._editDistance1(word)) or
                  self._known(self._editDistance2(word)) or
                  [word])

    return max(candidates, key=self.bagOfWords.get)


  def _known(self, words):
    """Return the subset of words that are in the corpus."""
    return {w for w in words if w in self.bagOfWords}


  @staticmethod
  def _editDistance1(word):
    """
    Return all strings that are edit distance =1 from the input word.
    Damerau-Levenshtein edit distance:
    - deletion(x,y) is the count(xy typed as x)
    - insertion(x,y) is the count(x typed as xy)
    - substitution(x,y) is the count(x typed as y)
    - transposition(x,y) is the count(xy typed as yx)
    """
    # First split the word into tuples of all possible pairs.
    # Note: need +1 so we can perform edits at front and back end of the word.
    splits = [(word[:i], word[i:]) for i in range(len(word)+1)]

    # Now perform the edits at every possible split location.
    # Substitution is essentially a deletion and insertion.
    delete = [a+b[1:] for a,b in splits if b]
    insert = [a+b+c for a,b in splits for c in TextPreprocess.alphabet]
    subs = [a+c+b[1:] for a,b in splits for c in TextPreprocess.alphabet if b]
    trans = [a+b[1]+b[0]+b[2:] for a,b in splits if len(b)>1]

    return set(delete + insert + subs + trans)


  def _editDistance2(self, word):
    """
    Return all strings that are edit distance =2 from the input word; i.e. call
    the _editDistance1() method twice for edits with distances of two.
    """
    return {edits2 for edits1 in self._editDistance1(word)
            for edits2 in self._editDistance1(edits1)}
