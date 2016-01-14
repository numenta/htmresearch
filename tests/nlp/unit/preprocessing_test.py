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

from htmresearch.support.text_preprocess import TextPreprocess



class TestTextPreprocess(unittest.TestCase):

  def setUp(self):
    self.preprocessor = TextPreprocess()
    self.testStrings = [
      "When you play GOT, you win or you die. There's no middl gruond.",
      "'A reader lives a thousand lives before he dies,' said Jojen. 'The man "
      "who never reads lives only one.'",
      "Hodor doesn't get to WFH."
    ]


  def testTokenization(self):
    tokenList = self.preprocessor.tokenize(self.testStrings[1])

    self.assertEqual(19, len(tokenList), "Incorrrect number of tokens.")
    self.assertNotIn("Jojen", tokenList, "Did not lower case words correctly.")
    self.assertNotIn([",",".","'"], tokenList, "Did not remove punctuation.")

    
  def testMapping(self):
    tokenList, mapping = self.preprocessor.tokenizeAndFilter(
      self.testStrings[0])
    
    # Test mapping a single word to multiple tokens.
    self.assertEqual(("There's", ["there", "s"]), mapping[9],
      "Didn't correctly map 'There's' to two tokens, 'there' and 's'.")
    
    # Test there is an entry for each original word.
    originalWords = self.testStrings[0].split(" ")
    for i, word in enumerate(originalWords):
      self.assertEqual(word, mapping[i][0],
        "The word '{}' is not mapped correctly.".format(word))
  

  def testContractionExpansion(self):
    tokenList, mapping = self.preprocessor.tokenizeAndFilter(
      self.testStrings[0], expandContr=True)

    self.assertEqual(("There's", ["there", "is"]), mapping[9],
      "Did not expand and map 'There's' as expected.")
  

  def testAbbreviationExpansion(self):
    tokenList, mapping = self.preprocessor.tokenizeAndFilter(
      self.testStrings[2], expandAbbr=True)

    self.assertEqual(("WFH.", ["work", "from", "home"]), mapping[4],
      "Did not expand and map 'WFH' as expected.")


  def testSpellCorrection(self):
    tokenList, _ = self.preprocessor.tokenizeAndFilter(
      self.testStrings[0], correctSpell=True)
    
    self.assertSequenceEqual(
      ["when", "you", "play", "got", "you", "win", "or", "you", "die", "there",
        "s", "no", "middle", "ground"],
      tokenList,
      "Did not correct the spelling as anticipated.")


  def testRemoveMostCommon(self):
    tokenList, _ = self.preprocessor.tokenizeAndFilter(
      self.testStrings[0], ignoreCommon=10)

    self.assertEqual(14, len(tokenList),
      "Text filtering shouldn't have removed any words.")

    tokenList, mapping = self.preprocessor.tokenizeAndFilter(
      self.testStrings[0], ignoreCommon=100)
 
    self.assertEqual(6, len(tokenList),
      "Didn't remove correct number of common words during text filtering.")
    
    # Test for words mapping to zero tokens.
    self.assertEqual(("no", []), mapping[10],
      "Didn't correctly map 'When' to an empty list of tokens.")


if __name__ == "__main__":
  unittest.main()
