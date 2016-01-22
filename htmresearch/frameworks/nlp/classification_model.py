# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2016, Numenta, Inc.  Unless you have purchased from
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

import cPickle as pkl
import numpy
import os
import shutil

from htmresearch.support.text_preprocess import TextPreprocess



class ClassificationModel(object):
  """
  Base class for NLP models of classification tasks. When inheriting from this
  class please take note of which methods MUST be overridden, as documented
  below.
  """

  def __init__(self,
               numLabels=None,
               verbosity=1,
               filterText=False,
               **kwargs):
    """
    @param verbosity  (int) Verbosity level. Larger values lead to more
                            printouts. 0 implies nothing will be printed.

    @param numLabels  (int) The maximum number of categories in the dataset.

    @param filterText (bool) Whether text will be filtered when tokenized.
                             Filtering may be model specific but by default
                             this includes ignoring common words, correcting
                             spelling, and removing the
                             string "[identified deleted]"
    """
    # TODO: we may want to provide more flexible filtering options, or even
    # an instance of the TextPreprocess class.

    if numLabels is None:
      raise RuntimeError("Must specify numLabels")

    self.numLabels = numLabels
    self.verbosity = verbosity
    self.filterText = filterText


  ################## CORE METHODS #####################

  def trainToken(self, token, labels, wordId, reset=0):
    """
    Train the model with the given text token, associated labels, and
    sampleId.

    @param token      (str)  The text token to train on
    @param labels     (list) A list of one or more integer labels associated
                             with this token.
    @param wordId     (int)  An integer ID associated with this token.
    @param reset      (int)  Should be 0 or 1. If 1, assumes we are at the
                             end of a sequence. A reset signal will be issued
                             after the model has been trained on this token.

    """
    raise NotImplementedError


  def trainDocument(self, document, labels, sampleId):
    """
    Train the model with the given document, associated labels, and sampleId.
    This routine will tokenize the document and train the model on these tokens
    using the given labels and id.  A reset will be issued after the
    document has been trained.

    @param document   (str)  The text token to train on.
    @param labels     (list) A list of one or more integer labels associated
                             with this token.
    @param sampleId   (int)  An integer ID associated with the entire document.
    """
    # Default implementation, may be overridden. Here we ignore the token-word
    # mappings produced by the tokenizer.
    assert (sampleId is not None), "Must pass in a sampleId"
    tokenList, _ = self.tokenize(document)
    lastTokenIndex = len(tokenList) - 1
    for i, token in enumerate(tokenList):
      self.trainToken(token, labels, sampleId, reset=int(i == lastTokenIndex))


  def inferToken(self, token, reset=0, returnDetailedResults=False,
                 sortResults=True):
    """
    Classify the token (i.e. run inference on the model with this document) and
    return classification results and (optionally) a list of sampleIds and
    distances.   Repeated sampleIds are NOT removed from the results.

    @param token    (str)     The text token to train on
    @param reset    (int)     Should be 0 or 1. If 1, assumes we are at the
                              end of a sequence. A reset signal will be issued
                              after the model has been trained on this token.
    @param returnDetailedResults
                    (bool)    If True will return sampleIds and distances
                              This could slow things down depending on the
                              number of stored patterns.
    @param sortResults (bool) If True the list of sampleIds and distances
                              will be sorted in order of increasing distances.

    @return  (numpy array) An array of size numLabels. Position i contains
                           the likelihood that this token belongs to the
                           i'th category. An array containing all zeros
                           implies no decision could be made.
             (list)        A list of sampleIds or None if
                           returnDetailedResults is False.
             (numpy array) An array of distances from each stored sample or
                           None if returnDetailedResults is False.
    """
    # TODO: Should we specify that distances normalized between 0 and 1?
    # Currently we use whatever the KNN returns.

    raise NotImplementedError


  def inferDocument(self, document, returnDetailedResults=False,
                    sortResults=True):
    """
    Run inference on the model with this document and return classification
    results, sampleIds and distances.  A reset is issued after inference.
    Repeated sampleIds ARE removed from the results.

    @param document (str)     The document to classify
    @param returnDetailedResults
                    (bool)    If True will return sampleIds and distances.
                              This could slow things down depending on the
                              number of stored patterns.
    @param sortResults (bool) If true the list of sampleIds and distances
                              will be sorted in order of increasing distances.

    @return  (numpy array) An array of size numLabels. Position i contains
                           the likelihood that this token belongs to the i'th
                           category. An array containing all zeros implies no
                           decision could be made.
             (list)        A list of unique sampleIds or None if
                           returnDetailedResults is False.
             (numpy array) An array of distances from each stored sample or
                           None if returnDetailedResults is False.
    """
    # TODO: normalize categoryVotes

    # Default implementation, can be overridden This default routine will
    # tokenize the document and classify using these tokens. The default
    # classification involves summing the most likely classification for each
    # token.

    # For each token run inference on the token and accumulate sum of distances
    # from this token to all other sampleIds.
    tokenList, _ = self.tokenize(document)

    if returnDetailedResults:
      return self._inferDocumentDetailed(tokenList, sortResults=sortResults)

    lastTokenIndex = len(tokenList) - 1
    categoryVotes = numpy.zeros(self.numLabels)

    for i, token in enumerate(tokenList):
      votes, _, _ = self.inferToken(token,
                                    reset=int(i == lastTokenIndex),
                                    returnDetailedResults=False,
                                    sortResults=False)

      if votes.sum() > 0:
        # Increment the most likely category, breaking ties in a random fashion
        sortedVotes = self._sortArray(votes)
        categoryVotes[sortedVotes[0]] += 1

    return categoryVotes, None, None


  def tokenize(self, inputText):
    """
    Given a bunch of text (could be several sentences) return a single list
    containing individual tokens. It will filterText if the global option
    is set.

    @param inputText  (str)   A bunch of text.
    @return sample    (list)  A list of text tokens.
    @return mapping   (list)  Maps the original words to the sample tokens. See
                              TextPreprocess method for details.
    """
    if self.filterText:
      sample, mapping = TextPreprocess().tokenizeAndFilter(inputText,
                                         ignoreCommon=100,
                                         removeStrings=["[identifier deleted]"],
                                         correctSpell=True)
    else:
      sample, mapping = TextPreprocess().tokenizeAndFilter(inputText)

    return sample, mapping


  def reset(self):
    """
    Issue a reset signal to the model. The assumption is that a sequence has
    just ended and a new sequence is about to begin.  The default behavior is
    to do nothing - not all subclasses may re-implement this.
    """
    pass


  ################## UTILITY METHODS #####################

  def setFilterText(self, filterText):
    """
    @param filterText (bool) If True, text will be filtered when tokenized.
    """
    self.filterText = filterText


  def getFilterText(self):
    return self.filterText


  def getClassifier(self):
    """
    Returns the classifier instance for the model.
    """
    raise NotImplementedError


  def dumpProfile(self):
    """
    Dump any profiling information. Subclasses can override this to provide
    custom profile reports.  Default is to do nothing.
    """
    pass


  def save(self, saveModelDir):
    """
    Save the model in the given directory.

    @param saveModelDir (string)
           Directory path for saving the model. This directory should only be
           used to store a saved model. If the directory does not exist, it will
           be created automatically and populated with model data. A
           pre-existing directory will only be accepted if it contains
           previously saved model data. If such a directory is given, the full
           contents of the directory will be deleted and replaced with current

    Implementation note: Subclasses should override _serializeExtraData() to
    save additional data in custom formats.
    """
    saveModelDir = os.path.abspath(saveModelDir)
    modelPickleFilePath = os.path.join(saveModelDir, "model.pkl")

    # Delete old model directory if we detect it
    if os.path.exists(saveModelDir):
      if (not os.path.isdir(saveModelDir) or
          not os.path.isfile(modelPickleFilePath) ):
        raise RuntimeError(("Existing filesystem entry <%s> is not a model"
                         " checkpoint -- refusing to delete"
                         " (%s missing or not a file)") %
                          (saveModelDir, modelPickleFilePath))

      shutil.rmtree(saveModelDir)

    # Create a new checkpoint directory for saving state
    self.__makeDirectoryFromAbsolutePath(saveModelDir)

    with open(modelPickleFilePath, "wb") as modelPickleFile:
      pkl.dump(self, modelPickleFile)

    # Tell the model to save extra data, if any.
    self._serializeExtraData(saveModelDir)


  @classmethod
  def load(cls, savedModelDir):
    """
    Create model from saved checkpoint directory and return it.

    @param savedModelDir (string)  Directory where model was saved

    @returns (ClassificationModel) The loaded model instance
    """
    savedModelDir = os.path.abspath(savedModelDir)
    modelPickleFilePath = os.path.join(savedModelDir, "model.pkl")

    with open(modelPickleFilePath, "rb") as modelPickleFile:
      model = pkl.load(modelPickleFile)

    # Tell the model to load extra data, if any.
    model._deSerializeExtraData(savedModelDir)

    return model


  @staticmethod
  def __makeDirectoryFromAbsolutePath(absDirPath):
    """
    Make directory for the given directory path if it doesn't already
    exist in the filesystem.

    @param absDirPath (string) Absolute path of the directory to create
    """

    assert os.path.isabs(absDirPath)

    # Create the experiment directory
    try:
      os.makedirs(absDirPath)
    except OSError as e:
      if e.errno != os.errno.EEXIST:
        raise


  def _serializeExtraData(self, extraDataDir):
    """
    Protected method that is called during serialization with an external
    directory path. It can be overridden by subclasses to save large binary
    states, bypass pickle, or for saving Network API instances.

    @param extraDataDir (string) Model's extra data directory path
    """
    pass


  def _deSerializeExtraData(self, extraDataDir):
    """
    Protected method that is called during deserialization (after __setstate__)
    with an external directory path. It can be overridden by subclasses to save
    large binary states, bypass pickle, or for saving Network API instances.

    @param extraDataDir (string) Model's extra data directory path
    """
    pass


  def _inferDocumentDetailed(self, tokenList, sortResults=True):
    """
    Run inference on the model with this list of tokens and return classification
    results, sampleIds and distances.  By default this routine will tokenize the
    document and classify using these tokens. A reset is issued after inference.
    Repeated sampleIds ARE removed from the results.

    @param tokenList (str)     The list of tokens for inference
    @param sortResults (bool) If true the list of sampleIds and distances
                              will be sorted in order of increasing distances.

    @return  (numpy array) An array of size numLabels. Position i contains
                           the likelihood that this token belongs to the i'th
                           category. An array containing all zeros implies no
                           decision could be made.
             (list)        Distances from this document to all prototypes in the
                           classifier, where each element is a 3-tuple:
                           (distance, unique ID, corresponding token index).
    """
    # Default implementation, can be overridden.

    # For each token in this document, run inference to get distances to all
    # prototypes in the classifier (depending on the model these may represent
    # documents or tokens), adding these distances to a cumulative sum of this
    # document's distances.

    lastTokenIndex = len(tokenList) - 1
    categoryVotes = numpy.zeros(self.numLabels)
    distancesForEachId = {}
    classifier = self.getClassifier()

    for i, token in enumerate(tokenList):
      votes, idList, distances = self.inferToken(token,
                                                 reset=int(i == lastTokenIndex),
                                                 returnDetailedResults=True,
                                                 sortResults=False)

      if votes.sum() > 0:
        if classifier.exact:
          # Increment all because a vote implies an exact match
          categoryVotes[numpy.where(votes > 0)[0]] = 1
          # We only care about 0 distances (exact matches), disregard all others
          distances[numpy.where(distances != 0)] = 1.0
        else:
          # Increment the most likely category, breaking ties in a random fashion
          sortedVotes = self._sortArray(votes)
          categoryVotes[sortedVotes[0]] += 1

        # For each prototype id (in the classifier), keep the minimum distance
        # to this inference token.
        for protoId in idList:
          # Find min distance of this protoId to this token
          closestDistance = distances[
            classifier.getPatternIndicesWithPartitionId(protoId)].min()

          # Add this to our running minimum of how close this protoId has
          # been to this document
          distancesForEachId[protoId] = (
            min(distancesForEachId.get(protoId, numpy.inf), closestDistance)
          )

    # Put distance from each prototype id to this document into a numpy array
    # ordered consistently with a list of protoIds
    protoIdList = distancesForEachId.keys()
    distanceToProtoIds = numpy.zeros(len(protoIdList))
    for i, protoId in enumerate(protoIdList):
      distanceToProtoIds[i] = distancesForEachId.get(protoId, numpy.inf)

    # Sort the results if requested
    if sortResults:
      sortedIndices = distanceToProtoIds.argsort()
      sortedDistances = distanceToProtoIds[sortedIndices]
      sortedIdList = [protoIdList[i] for i in sortedIndices]

      return categoryVotes, sortedIdList, sortedDistances

    else:
      return categoryVotes, protoIdList, distanceToProtoIds


  @staticmethod
  def _sortArray(array, seed=42):
    """
    Sort the input array, breaking ties in a random fashion.
    @param array (numpy array)    Array of values to be sorted.
    @param seed (int)             Seed the random number generator.
    @return   (numpy array)       Sorted array indices, where the sort order is
                                  greatest to least.
    """
    numpy.random.seed(seed)
    randomValues = numpy.random.random(array.size)
    return numpy.lexsort((randomValues, array))[::-1]
