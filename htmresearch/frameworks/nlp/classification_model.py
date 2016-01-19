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

  def trainToken(self, token, labels, sampleId, reset=0):
    """
    Train the model with the given text token, associated labels, and
    sampleId.

    @param token      (str)  The text token to train on
    @param labels     (list) A list of one or more integer labels associated
                             with this token.
    @param sampleId   (int)  An integer ID associated with this token.
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

    @param document   (str)  The text token to train on
    @param labels     (list) A list of one or more integer labels associated
                             with this token.
    @param sampleId   (int)  An integer ID associated with the entire document.

    """
    # Default implementation, may be overridden
    assert (sampleId is not None), "Must pass in a sampleId"
    tokenList = self.tokenize(document)
    lastTokenIndex = len(tokenList) - 1
    for i,token in enumerate(tokenList):
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
    # Default implementation, can be overridden This default routine will
    # tokenize the document and classify using these tokens. The default
    # classification involves summing the most likely classification for each
    # token.

    # For each token run inference on the token and accumulate sum of distances
    # from this token to all other sampleIds.
    tokenList = self.tokenize(document)

    lastTokenIndex = len(tokenList) - 1
    categoryVotes = numpy.zeros(self.numLabels)

    if returnDetailedResults:
      return self._inferDocumentDetailed(tokenList, sortResults=sortResults)

    for i, token in enumerate(tokenList):
      votes, _, _ = self.inferToken(token,
                                     reset=int(i == lastTokenIndex),
                                     returnDetailedResults=False,
                                     sortResults=False)

      if votes.sum() > 0:
        categoryVotes[votes.argmax()] += 1

    return categoryVotes, None, None


  def tokenize(self, preprocessedText):
    """
    Given a bunch of text (could be several sentences) return a single list
    containing individual tokens.  It will filterText if the global option
    is set.

    @param preprocessedText  (str)     A bunch of text.
    @return                  (list)    A list of text tokens.

    """
    if self.filterText:
      sample = TextPreprocess().tokenize(preprocessedText,
                                         ignoreCommon=100,
                                         removeStrings=["[identifier deleted]"],
                                         correctSpell=True)
    else:
      sample = TextPreprocess().tokenize(preprocessedText)

    return sample


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
             (list)        A list of unique sampleIds
             (numpy array) An array of distances from this document to each
                           sampleId
    """
    # Default implementation, can be overridden.
    # Note that some models specify a classifier with exact matching, which is
    # reflected in the returned categoryVotes, but not the returned distances.

    # For each token run inference on the token and accumulate sum of distances
    # from this token to all other sampleIds.
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
        categoryVotes[votes.argmax()] += 1

        # For each unique id, keep the minimum distance to this token
        for j, sampleId in enumerate(idList):
          # Find min distance of this sampleId to this token
          closestDistance = distances[
            classifier.getPatternIndicesWithPartitionId(sampleId)].min()

          # Add this to our running minimum of how close this sampleId has
          # been to this document
          distancesForEachId[sampleId] = (
            min(distancesForEachId.get(sampleId, numpy.inf), closestDistance)
          )


    # Put distance from each sampleId to this document into a numpy array
    # ordered consistently with a list of sampleIds
    sampleIdList = distancesForEachId.keys()
    distancetoSampleIds = numpy.zeros(len(sampleIdList))
    for i,sampleId in enumerate(sampleIdList):
      distancetoSampleIds[i] = distancesForEachId.get(sampleId, numpy.inf)

    # Sort the results if requested
    if sortResults:
      sortedIndices = distancetoSampleIds.argsort()
      sortedDistances = distancetoSampleIds[sortedIndices]
      sortedIdList = []
      for i in sortedIndices:
        sortedIdList.append(sampleIdList[i])

      return categoryVotes, sortedIdList, sortedDistances

    else:
      return categoryVotes, sampleIdList, distancetoSampleIds
