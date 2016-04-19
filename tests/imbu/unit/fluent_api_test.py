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

from paste.fixture import TestApp
import simplejson as json
import unittest

import fluent_api



class TestFluentAPI(unittest.TestCase):

  def setUp(self):
    self.app = TestApp(fluent_api.app.wsgifunc())


  def _assertCORSHeaders(self, response):
    self.assertIn("access-control-allow-headers", response.header_dict)
    self.assertEqual(response.header_dict["access-control-allow-headers"],
                     "accept, access-control-allow-origin, content-type")
    self.assertIn("access-control-allow-credentials", response.header_dict)
    self.assertEqual(response.header_dict["access-control-allow-credentials"],
                     "true")
    self.assertIn("access-control-allow-headers", response.header_dict)
    self.assertEqual(response.header_dict["access-control-allow-origin"], "*")
    self.assertIn("access-control-allow-headers", response.header_dict)
    self.assertEqual(response.header_dict["access-control-allow-methods"],
                     "POST")


  def _queryAndAssertAPI(self, model=None, dataset=None, query=""):
    uri = "/fluent"
    if model:
      uri = "{}/{}".format(uri, model)
      if dataset:
        uri = "{}/{}".format(uri, dataset)
    response = self.app.post(uri, json.dumps(query))

    self.assertEqual(response.status, 200)
    self._assertCORSHeaders(response)

    # Make sure the response can be parsed as JSON
    body = json.loads(response.body)

    return body


  def testDefaultResponse(self):
    responseBody = self._queryAndAssertAPI()

    # Assert structure of response matches expected pattern
    for _, result in responseBody.iteritems():
      self.assertIn("text", result)
      self.assertIn("scores", result)
      # Assert scores=0 b/c no query data sent
      self.assertEqual(result["scores"], [0])


  def testQueryAModel(self):
    # Should use default dataset as declared in ImbuModels. Assume a
    # corresponding serialized CioDocumentFingerprint model exists.
    responseBody = self._queryAndAssertAPI(model="CioDocumentFingerprint")

    # Assert structure of response matches expected pattern
    for _, result in responseBody.iteritems():
      self.assertIn("text", result)
      self.assertIn("scores", result)
      self.assertIn("windowSize", result)


  def testQueryWithoutAModel(self):
    # Should use default model and dataset as declared in ImbuModels
    responseBody = self._queryAndAssertAPI(query="test")

    # Assert structure of response matches expected pattern
    for _, result in responseBody.iteritems():
      self.assertIn("text", result)
      self.assertIn("scores", result)
      self.assertIn("windowSize", result)


  def testDatasetList(self):
    response = self.app.get("/fluent/datasets")

    self.assertEqual(response.status, 200)
    self._assertCORSHeaders(response)

    # Make sure the response can be parsed as JSON
    datasets = json.loads(response.body)

    # Assert structure of response matches expected pattern
    self.assertIsInstance(datasets, list)
    self.assertGreaterEqual(len(datasets), 1)


  def testQueryADatasetModel(self):
    # Assume 'sample_reviews' dataset exists
    responseBody = self._queryAndAssertAPI(dataset="sample_reviews",
                                           model="CioDocumentFingerprint",
                                           query="test")

    # Assert structure of response matches expected pattern
    for _, result in responseBody.iteritems():
      self.assertIn("text", result)
      self.assertGreater(len(result["text"]), 0)
      self.assertIn("scores", result)
      self.assertIn("windowSize", result)
