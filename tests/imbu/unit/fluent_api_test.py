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

import unittest
import simplejson as json
from paste.fixture import TestApp

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


  def testDefaultResponse(self):
    response = self.app.post("/fluent")
    self.assertEqual(response.status, 200)
    self._assertCORSHeaders(response)

    # Assert that response can be parsed as JSON
    body = json.loads(response.body)

    # Assert structure of response matches expected pattern
    for _, result in body.iteritems():
      self.assertIn("text", result)
      self.assertIn("scores", result)
      self.assertEqual(result["scores"], [0])


  def _queryModelAndAssertResponse(self, model=None, query="test"):
    uri = "/fluent"
    if model:
      uri = "{}/{}".format(uri, model)

    response = self.app.post(uri, json.dumps(query))
    self.assertEqual(response.status, 200)
    self._assertCORSHeaders(response)

    # Assert that response can be parsed as JSON
    body = json.loads(response.body)

    # Assert structure of response matches expected pattern
    for _, result in body.iteritems():
      self.assertIn("text", result)
      self.assertIn("scores", result)
      self.assertIn("windowSize", result)


  def testQueryCioWordFingerprint(self):
    self._queryModelAndAssertResponse("CioWordFingerprint")


  def testQueryCioDocumentFingerprint(self):
    self._queryModelAndAssertResponse("CioDocumentFingerprint")


  def testQueryHTMSensorKnn(self):
    self._queryModelAndAssertResponse("HTM_sensor_knn")


  @unittest.skip("IMBU-101 need to retrain models to overcome backwards "
                 "incompatibility")
  def testQueryHTMSensorSimpleTpKnn(self):
    self._queryModelAndAssertResponse("HTM_sensor_simple_tp_knn")


  def testQueryKeywords(self):
    self._queryModelAndAssertResponse("Keywords")
