#!/usr/bin/env python
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

"""Tests for the TextPreprocess class."""

import unittest

from htmresearch.support.text_preprocess import TextPreprocess



class TextPreprocessTest(unittest.TestCase):


  def testTokenizeNoPreprocess(self):
    """Tests none of the preprocessing methods are used."""
    text = "I can't work at [identifier deleted] if you don't allw me to wfh"
    processor = TextPreprocess()

    expected_tokens = ["i", "can", "t", "work", "at", "identifier", "deleted",
                       "if", "you", "don", "t", "allw", "me", "to", "wfh"]
    tokens = processor.tokenize(text)
    self.assertSequenceEqual(tokens, expected_tokens)


  def testTokenizeRemoveString(self):
    """Tests a provided string is ignored."""
    text = "I can't work at [identifier deleted] if you don't allw me to wfh"
    processor = TextPreprocess()

    expected_tokens = ["i", "can", "t", "work", "at", "if", "you", "don",
                       "t", "allw", "me", "to", "wfh"]
    tokens = processor.tokenize(text, removeStrings=["[identifier deleted]"])
    self.assertSequenceEqual(tokens, expected_tokens)


  def testTokenizeExpandAbbreviation(self):
    """Tests abbreviations are expanded."""
    text = "I can't work at [identifier deleted] if you don't allw me to wfh"
    processor = TextPreprocess()

    expected_tokens = ["i", "can", "t", "work", "at", "identifier", "deleted",
                       "if", "you", "don", "t", "allw", "me", "to", "work",
                       "from", "home"]
      
    tokens = processor.tokenize(text, expandAbbr=True)
    self.assertSequenceEqual(tokens, expected_tokens)


  def testTokenizeExpandContraction(self):
    """Tests contractions are expanded."""
    text = "I can't work at [identifier deleted] if you don't allw me to wfh"
    processor = TextPreprocess()

    expected_tokens = ["i", "can", "not", "work", "at", "identifier", "deleted",
                       "if", "you", "do", "not", "allw", "me", "to", "wfh"]
    tokens = processor.tokenize(text, expandContr=True)
    self.assertSequenceEqual(tokens, expected_tokens)


  def testFunctionsWithoutDataFiles(self):
    """
    Ensures a TextPreprocess object can be created and tokenize when there are
    no text data files (corpus text, abbreviations, and contractions).
    """
    text = "I can't work at [identifier deleted] if you don't allw me to wfh"
    processor = TextPreprocess(corpusTxt="fake.txt",
                               abbrCSV="not_here.csv",
                               contrCSV="not_real.csv")
    
    tokens = processor.tokenize(text)
    expected_tokens = ["i", "can", "t", "work", "at", "identifier", "deleted",
                       "if", "you", "don", "t", "allw", "me", "to", "wfh"]
    
    self.assertSequenceEqual(tokens, expected_tokens)


  def testReadExpansionFileNoSuffixes(self):
    """Tests TextPreprocess reads csv files correctly."""
    processor = TextPreprocess()
    abbreviations = processor.readExpansionFile("abbreviations.csv")
    expectedAbbreviations = {"wfh": "work from home"}
    self.assertEqual(abbreviations, expectedAbbreviations)

  
  def testReadExpansionFileWithSuffixes(self):
    """Tests TextPreprocess reads csv files correctly and adds suffixes."""
    processor = TextPreprocess()
    suffixes = ["", "s", "'s"]
    abbreviations = processor.readExpansionFile("abbreviations.csv", suffixes)
    expectedAbbreviations = {"wfh": "work from home",
                             "wfhs": "work from homes",
                             "wfh's": "work from home's"}
    self.assertEqual(abbreviations, expectedAbbreviations)


if __name__ == "__main__":
  unittest.main()
