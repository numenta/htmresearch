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

from htmresearch.frameworks.nlp.imbu import ImbuModels
from htmresearch.frameworks.nlp.model_factory import ClassificationModelTypes



class TestImbu(unittest.TestCase):

  def setUp(self):
    root = (
      os.path.dirname(
        os.path.dirname(
          os.path.dirname(
            os.path.dirname(
              os.path.realpath(__file__)
            )
          )
        )
      )
    )
    self.dataPath = os.path.join(
      root, "projects/nlp/data/sample_reviews/sample_reviews.csv")


  def testMappingModelNamesToModelTypes(self):
    imbu = ImbuModels(cacheRoot="fake_cache_root", dataPath=self.dataPath)

    for modelName, mappedName in imbu.modelMappings.iteritems():
      self.assertEquals(mappedName, imbu._mapModelName(modelName),
        "Incorrect mapping returned for model named '{}'".format(modelName))

    self.assertRaises(ValueError, imbu._mapModelName, "fakeModel")



if __name__ == "__main__":
  unittest.main()
