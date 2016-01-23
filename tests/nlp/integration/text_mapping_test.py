#!/usr/bin/env python
# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2016, Numenta, Inc.  Unless you have purchased from
# Numenta, Inc. a separate commercial license for this software code, the
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

import os
import unittest

from htmresearch.encoders import EncoderTypes
from htmresearch.frameworks.nlp.model_factory import createModel
from htmresearch.support.text_preprocess import TextPreprocess



class TestTextPreprocess(unittest.TestCase):

  tokenIndexingFactor = 1000
  documentLevel = {"CioDocumentFingerprint", "CioWordFingerprint"}


  def setUp(self):
    self.testDocuments = (
      ("Much of the world's data is streaming, time-series data, where "
       "anomalies give significant information in critical situations; "
       "examples abound in domains such as finance, IT, security, medical, "
       "and energy. Yet detecting anomalies in streaming data is a difficult "
       "task, requiring detectors to process data in real-time, not batches, "
       "and learn while simultaneously making predictions... The goal for "
       "[identifier deleted] is to provide a standard, open source framework "
       "with which the research community can compare and evaluate different "
       "algorithms for detecting anomalies in streaming data."),
      ("We propose a formal mathematical model for sparse representation in "
       "neocortex based on a neuron model and associated operations... As such "
       "the theory provides a unified and practical mathematical framework for "
       "understanding the benefits and limits of sparse representation in "
       "cortical networks."),
      ("Therefor the HTM sequence memory doesn't only advance our "
       "understanding of how the brain may solve the sequence learning "
       "problem, but it's applicable to a wide range of real-world problems "
       "such as dicsrete and continuous sequence prediction, anomaly "
       "detection, and sequence classification."),
      ("In this paper we extnd this idea by showing that a neuron with several "
       "thousand synapses aranged along active dendrites can learn to "
       "accurately and robustly recognize hundreds of unique patterns of "
       "cellular activity, even in the presence of large amounts of noise and "
       "pattern variation... Thus neurons need thousands of synapses to learn "
       "the many temporal patterns in sensory stimuli and motor sequence."),
      )

    self.filteredProtoIds = ( [0, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17,
      18, 20, 23, 25, 26, 28, 29, 30, 31, 33, 34, 37, 38, 39, 40, 42, 43, 45,
      47, 49, 50, 51, 52, 53, 55, 57, 58, 61, 63, 64, 65, 66, 70, 71, 72, 73,
      75, 76, 77, 79, 80, 82, 83, 1001, 1003, 1004, 1005, 1007, 1008, 1010,
      1011, 1014, 1015, 1017, 1018, 1022, 1023, 1025, 1027, 1028, 1029, 1031,
      1033, 1035, 1037, 1038, 1040, 1041, 2000, 2002, 2003, 2004, 2005, 2007,
      2008, 2009, 2013, 2015, 2017, 2018, 2019, 2022, 2025, 2026, 2028, 2029,
      2032, 2034, 2035, 2036, 2037, 2038, 2040, 2041, 3002, 3004, 3006, 3008,
      3011, 3013, 3014, 3015, 3016, 3017, 3018, 3019, 3020, 3021, 3023, 3025,
      3026, 3027, 3029, 3030, 3032, 3033, 3034, 3037, 3039, 3040, 3042, 3044,
      3045, 3046, 3047, 3048, 3049, 3051, 3053, 3055, 3056, 3057, 3059, 3060,
      3062, 3063] )


  def _formatResults(self, model, modelName, distanceArray, idList):
    """ As implemented in imbu.py: Format distances to reflect the
    pctOverlapOfInput metric, return a list of results.
    """
    formattedDistances = (1.0 - distanceArray) * 100

    results = []
    for protoId, dist in zip(idList, formattedDistances):
      if modelName in self.documentLevel:
        results.append({"sampleId": protoId,
                        "wordId": 0,
                        "text": self.testDocuments[protoId],
                        "score": dist.item()})
      else:
        # get the sampleId from the protoId
        wordId = protoId % self.tokenIndexingFactor
        sampleId = (protoId - wordId) / self.tokenIndexingFactor
        results.append({"sampleId": sampleId,
                        "wordId": wordId,
                        "text": self.testDocuments[sampleId],
                        "score": dist.item()})

    return results


  def testMappingsWithImbuWordModel(self):
    # Create a Keywords model
    modelName = "Keywords"
    kwargs = {"numLabels": 1,
              "k": 42,
              "classifierMetric": "pctOverlapOfInput",
              "filterText": True,
              "verbosity": 0}
    model = createModel(modelName, **kwargs)

    # Train the model for use in Imbu
    for seqId, text in enumerate(self.testDocuments):
      tokenList, mapping = model.tokenize(text)
      lastTokenIndex = len(tokenList) - 1
      for i, (token, tokenIndex) in enumerate(zip(tokenList, mapping)):
        wordId = seqId * self.tokenIndexingFactor + tokenIndex
        model.trainToken(token,
                         [0],
                         wordId,
                         reset=int(i == lastTokenIndex))

    # Query the model, expecting two matches from one sample
    query = ("The key to artificial intelligence has always been the "
             "representation.")
    _, sortedIds, sortedDistances = model.inferDocument(
      query, returnDetailedResults=True, sortResults=True)

    # Test for expected word-token mapping (in prototype IDs)
    self.assertItemsEqual(self.filteredProtoIds, sortedIds,
      "List of IDs returned from inference does not match the expected list of "
      "prototype IDs.")

    # Test for exact matching results
    self.assertSequenceEqual([0.0, 0.0, 1.0], sortedDistances[:3].tolist(),
      "Expected two exact-matching prototypes.")

    # Test for multiple matches per sample
    results = self._formatResults(model, modelName, sortedDistances, sortedIds)
    self.assertEqual(results[0]["sampleId"], results[1]["sampleId"])
    self.assertEqual(results[0]["text"], results[1]["text"])
    self.assertNotEqual(results[0]["wordId"], results[1]["wordId"])

    # Test the match maps back to the query
    matchingWord = results[0]["text"].split(" ")[results[0]["wordId"]]
    self.assertIn(matchingWord, query, "Matching word is indexed incorrectly.")

    # Query the model again, expecting five matches from two samples
    query = ("sequence")
    _, sortedIds, sortedDistances = model.inferDocument(
      query, returnDetailedResults=True, sortResults=True)

    # Test for exact matching results
    self.assertSequenceEqual(
      [0.0, 0.0, 0.0, 0.0, 0.0, 1.0], sortedDistances[:6].tolist(),
      "Expected five exact-matching prototypes.")

    # Test the exact matches map back to the query term
    results = self._formatResults(model, modelName, sortedDistances, sortedIds)
    for r in results[:5]:
      self.assertIn(r["sampleId"], (2,3))
      matchingWord = r["text"].split(" ")[r["wordId"]]
      self.assertIn(query, matchingWord,
        "Matching word is indexed incorrectly.")


  def testMappingsWithImbuDocumentModel(self):
    # Create the CioDocumentFingerprint model
    modelName = "CioDocumentFingerprint"
    kwargs = {"numLabels": 1,
              "classifierMetric": "pctOverlapOfInput",
              "filterText": True,
              "verbosity": 0,
              "fingerprintType": EncoderTypes.document,
              "cacheRoot": None}
    model = createModel("CioDocumentFingerprint", **kwargs)

    # Train the model for use in Imbu
    for seqId, text in enumerate(self.testDocuments):
      model.trainDocument(text, [0], seqId)

    # Query the model, expecting two matches from one sample
    query = ("The key to artificial intelligence has always been the "
             "representation.")
    _, sortedIds, sortedDistances = model.inferDocument(
      query, returnDetailedResults=True, sortResults=True)

    self.assertEqual(len(self.testDocuments), len(sortedIds),
      "Document-level models should have one prototype ID per document.")

    results = self._formatResults(model, modelName, sortedDistances, sortedIds)

    for r in results:
      self.assertEqual(0, r["wordId"],
        "wordId is insignificant in document-level models, and should be 0.")


  def testIndexMapping(self):
    originalWords = self.testDocuments[2].split(" ")

    tokenList, mapping = TextPreprocess().tokenizeAndFilter(
        self.testDocuments[2],
        ignoreCommon=50,
        removeStrings=["[identifier deleted]"],
        correctSpell=True,
        expandAbbr=True,
        expandContr=True)

    self.assertEqual(len(tokenList), len(mapping),
      "There should be one mapping entry for each token.")

    # Test filtering results
    self.assertEqual("therefore", tokenList[0], "Spelling not corrected.")
    self.assertEqual("discrete", tokenList[24], "Spelling not corrected.")
    self.assertSequenceEqual(["hierarchical", "temporal", "memory"],
      tokenList[1:4], "Abbreviation 'HTM' not expanded.")
    self.assertNotIn("but", tokenList, "Common word 'but' not removed.")
    self.assertNotIn("not", tokenList, "Common word 'not' not removed.")
    self.assertIn("does", tokenList, "Contraction not expanded to 'does not'.")


    # Test some token-to-word-mappings
    mappedWords = [originalWords[i] for i in mapping]
    self.assertNotEqual(len(originalWords), len(mappedWords))
    for word in mappedWords[1:4]:
      self.assertEqual("HTM", word,
        "Tokens don't map to 'HTM' as expected.")



if __name__ == "__main__":
  unittest.main()
