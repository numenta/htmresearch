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

from htmresearch.encoders import EncoderTypes
from htmresearch.encoders.cio_encoder import CioEncoder

try:
  import simplejson as json
except ImportError:
  import json


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
  
  
  def testDocumentFingerprint(self):
    """Test the Cortical.io text (document-level) encoding."""

    cio = CioEncoder(fingerprintType=EncoderTypes.document)
    response = cio.encode(self.text)
    
    self.assertFingerprintFields(response)
    
    encodingDict = getTestData("cio_encoding_document.json")
    
    self.assertEqual(encodingDict["fingerprint"]["positions"],
      response["fingerprint"]["positions"], "Cio bitmap is not as expected.")


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
    
    cio = CioEncoder(retinaScaling = 0.5, fingerprintType=EncoderTypes.document)
    response = cio.encode(self.text)
    
    encodingDict = getTestData("cio_encoding_scaled_retina.json")

    self.assertEqual(encodingDict["fingerprint"]["positions"],
      response["fingerprint"]["positions"], "Cio bitmap is not as expected.")

    fullRetinaEncodingDict = getTestData("cio_encoding_document.json")
    fullLength = len(fullRetinaEncodingDict["fingerprint"]["positions"])
    responseLength = len(response["fingerprint"]["positions"])
    
    self.assertTrue(responseLength <= fullLength,
      "Retina scaling did not decrease the fingerprint size.")



if __name__ == "__main__":
     unittest.main()
