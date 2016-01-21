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

import os
import unittest
import hashlib

import simplejson as json

from htmresearch.encoders import EncoderTypes
from htmresearch.encoders.cio_encoder import CioEncoder
from htmresearch.support.text_preprocess import TextPreprocess



DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_data")


def getTestData(fileName):
  with open(os.path.join(DATA_DIR, fileName)) as dataFile:
    return json.load(dataFile)



class CioTest(unittest.TestCase):
  """Test class for getting Cortical.io encodings."""

  def setUp(self):
    self.text = "Beautiful is better than ugly."

  def assertFingerprintFields(self, response):
    """Check we get the expected fields in a fingerprint dict."""

    self.assertIsInstance(response["text"], str,
      "Text field is not a string.")
    self.assertIsInstance(response["fingerprint"]["positions"], list,
      "Positions field is not a list.")
    self.assertIsInstance(response["sparsity"], float,
      "Sparsity field is not a float.")
    self.assertIsInstance(response["width"], int,
      "Width field is not an int.")
    self.assertIsInstance(response["height"], int,
      "Height field is not an int.")


  @unittest.skip("needs to be fixed.")
  def testDocumentFingerprint(self):
    """Test the Cortical.io text (document-level) encoding."""

    cio = CioEncoder(fingerprintType=EncoderTypes.document)
    response = cio.encode(self.text)

    self.assertFingerprintFields(response)

    encodingDict = getTestData("cio_encoding_document.json")

    self.assertEqual(encodingDict["fingerprint"]["positions"],
      response["fingerprint"]["positions"], "Cio bitmap is not as expected.")


  @unittest.skip("needs to be fixed.")
  def testWordFingerprint(self):
    """Test the Cortical.io term (word-lelevl) encoding."""

    cio = CioEncoder(fingerprintType=EncoderTypes.word)
    response = cio.encode(self.text)

    self.assertFingerprintFields(response)

    encodingDict = getTestData("cio_encoding_word.json")

    self.assertEqual(encodingDict["fingerprint"]["positions"],
      response["fingerprint"]["positions"], "Cio bitmap is not as expected.")


  def testRetinaScaling(self):
    """Test the CioEncoder for retina dimension scaling."""

    cio = CioEncoder(
      retinaScaling = 1.0, fingerprintType=EncoderTypes.document)
    cioScaled = CioEncoder(
      retinaScaling = 0.5, fingerprintType=EncoderTypes.document)

    self.assertAlmostEqual(0.5*cio.width, cioScaled.width)
    self.assertAlmostEqual(0.5*cio.height, cioScaled.height)

    response = cio.encode(self.text)
    responseScaled = cioScaled.encode(self.text)

    # Each bit position should be scaled down by retinaScaling*retinaScaling
    self.assertLessEqual(responseScaled["fingerprint"]["positions"].sum(),
                         0.5*0.5*response["fingerprint"]["positions"].sum())

    # The number of on bits in scaled retina should normally be slightly less
    # than the original, but can be equal in some cases
    self.assertLessEqual(len(responseScaled["fingerprint"]["positions"]),
                         len(response["fingerprint"]["positions"]))


  def testMaxSparsity(self):
    """Test that CioEncoder's maxSparsity works."""

    # This text seems to generate bitmaps with about 8% sparsity
    text = ("Smoking harms nearly every organ in your body. Over 7000 chemicals"
            " have been identified in tobacco smoke. After reading all this"
            " James and Sue decided to abruptly quit cigarette smoking to"
            " improve their health but it clearly was not an easy decision.")

    # Encoders with maxSparsity of 100%, 10%, 5%, and 1%
    cio100 = CioEncoder(maxSparsity=1.0, fingerprintType=EncoderTypes.document)
    cio10 = CioEncoder(maxSparsity=0.1, fingerprintType=EncoderTypes.document)
    cio5 = CioEncoder(maxSparsity=0.05, fingerprintType=EncoderTypes.document)
    cio1 = CioEncoder(maxSparsity=0.01, fingerprintType=EncoderTypes.document)

    bitmapSize = cio100.width*cio100.height
    r100 = cio100.encode(text)
    r10 = cio10.encode(text)
    r5 = cio5.encode(text)
    r1 = cio1.encode(text)

    length100 = len(r100["fingerprint"]["positions"])
    length10 = len(r10["fingerprint"]["positions"])
    length5 = len(r5["fingerprint"]["positions"])
    length1 = len(r1["fingerprint"]["positions"])

    # Encodings must have no more than desired sparsity
    self.assertLessEqual(r100["sparsity"], 1.0)
    self.assertLessEqual(r10["sparsity"], 0.1)
    self.assertLessEqual(r5["sparsity"], 0.05)
    self.assertLessEqual(r1["sparsity"], 0.01)

    self.assertLessEqual(length100, bitmapSize)
    self.assertLessEqual(length10, 0.1*bitmapSize)
    self.assertLessEqual(length5, 0.05*bitmapSize)
    self.assertLessEqual(length1, 0.01*bitmapSize)

    # Encodings can't be zero
    self.assertGreater(length100, 0)
    self.assertGreater(length10, 0)
    self.assertGreater(length5, 0)
    self.assertGreater(length1, 0)

    # Encodings must have complete overlap with the next higher encoding
    s100 = set(r100["fingerprint"]["positions"])
    s10 = set(r10["fingerprint"]["positions"])
    s5 = set(r5["fingerprint"]["positions"])
    s1 = set(r1["fingerprint"]["positions"])
    self.assertEqual(len(s100 & s10), length10)
    self.assertEqual(len(s10 & s5), length5)
    self.assertEqual(len(s5 & s1), length1)

    # Test that if you encode a second time, you get the same bitmap
    r100_2 = cio100.encode(text)
    r10_2 = cio10.encode(text)
    r5_2 = cio5.encode(text)
    r1_2 = cio1.encode(text)

    self.assertEqual(hashlib.sha224(str(r100)).hexdigest(),
                      hashlib.sha224(str(r100_2)).hexdigest())
    self.assertEqual(hashlib.sha224(str(r10)).hexdigest(),
                      hashlib.sha224(str(r10_2)).hexdigest())
    self.assertEqual(hashlib.sha224(str(r5)).hexdigest(),
                      hashlib.sha224(str(r5_2)).hexdigest())
    self.assertEqual(hashlib.sha224(str(r1)).hexdigest(),
                      hashlib.sha224(str(r1_2)).hexdigest())



  def testWindowEncodings(self):
    """Test the CioEncoder for the sliding window encodings."""
    cio = CioEncoder(fingerprintType=EncoderTypes.word)

    text = """
      I grok people. I am people, so now I can say it in people talk. I've found
      out why people laugh. They laugh because it hurts so much, because it's
      the only thing that'll make it stop hurting."""

    tokens = TextPreprocess().tokenize(text)

    encodingDicts = cio.getWindowEncoding(tokens, minSparsity=0.19)
    
    # Test that only dense windows get encoded
    self.assertTrue(len(tokens) > len(encodingDicts),
      "Returned incorrect number of window encodings.")

    # Test window
    windowEncoding = getTestData("cio_encoding_window.json")
    self.assertEqual(windowEncoding["text"], encodingDicts[-1]["text"],
      "Window encoding represents the wrong text.")
    self.assertTrue(encodingDicts[-1]["sparsity"] <= cio.unionSparsity,
      "Sparsity for large window is larger than the max.")
    self.assertSequenceEqual(
      windowEncoding["bitmap"], encodingDicts[-1]["bitmap"].tolist(),
      "Window encoding's bitmap is not as expected.")



if __name__ == "__main__":
    unittest.main()
