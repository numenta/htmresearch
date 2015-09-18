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

"""Tests for the NetworkDataGenerator class."""

import os
import pandas
import random
import unittest

from fluent.utils.network_data_generator import NetworkDataGenerator
from nupic.data.file_record_stream import FileRecordStream

try:
  import simplejson as json
except:
  import json



class NetworkDataGeneratorTest(unittest.TestCase):


  def __init__(self, *args, **kwargs):
    super(NetworkDataGeneratorTest, self).__init__(*args, **kwargs)
    self.expected = [[
      {"_token": "get",
      "_categories": "0 1",
      "_sequenceID": 0,
      "ID": "1",
      "_reset": 1},
      {"_token": "rid",
      "_categories": "0 1",
      "_sequenceID": 0,
      "ID": "1",
      "_reset": 0},
      {"_token": "of",
      "_categories": "0 1",
      "_sequenceID": 0,
      "ID": "1",
      "_reset": 0},
      {"_token": "the",
      "_categories": "0 1",
      "_sequenceID": 0,
      "ID": "1",
      "_reset": 0},
      {"_token": "trrible",
      "_categories": "0 1",
      "_sequenceID": 0,
      "ID": "1",
      "_reset": 0},
      {"_token": "kitchen",
      "_categories": "0 1",
      "_sequenceID": 0,
      "ID": "1",
      "_reset": 0},
      {"_token": "odor",
      "_categories": "0 1",
      "_sequenceID": 0,
      "ID": "1",
      "_reset": 0}],
      [{"_token": "i",
      "_categories": "2",
      "_sequenceID": 1,
      "ID": "2",
      "_reset": 1},
      {"_token": "don",
      "_categories": "2",
      "_sequenceID": 1,
      "ID": "2",
      "_reset": 0},
      {"_token": "t",
      "_categories": "2",
      "_sequenceID": 1,
      "ID": "2",
      "_reset": 0},
      {"_token": "care",
      "_categories": "2",
      "_sequenceID": 1,
      "ID": "2",
      "_reset": 0}]]
    self.dirName = os.path.dirname(os.path.realpath(__file__))


  def assertRecordsEqual(self, actual, expected):
    self.assertIsInstance(actual, list)
    self.assertEqual(len(actual), len(expected))
    for a, e in zip(actual, expected):
      self.assertEqual(len(a), len(e))
      for ra, re in zip(a, e):
        self.assertDictEqual(ra, re)


  def testSplitNoPreprocess(self):
    ndg = NetworkDataGenerator()
    filename = os.path.join(self.dirName, "test_data/multi_sample.csv")

    ndg.split(filename, 3, False)
    self.assertRecordsEqual(ndg.records, self.expected)

  
  def testSplitPreprocess(self):
    ndg = NetworkDataGenerator()
    filename = os.path.join(self.dirName, "test_data/multi_sample.csv")

    expected = [[
      {"_token": "gohbkchoo",
      "_categories": "0 1",
      "_sequenceID": 0,
      "ID": "1",
      "_reset": 1}],
      [{"_token": "o",
      "_categories": "2",
      "_sequenceID": 1,
      "ID": "2",
      "_reset": 1},
      {"_token": "ca",
      "_categories": "2",
      "_sequenceID": 1,
      "ID": "2",
      "_reset": 0}]]

    ndg.split(filename, 3, True, ignoreCommon=100, correctSpell=True)
    self.assertRecordsEqual(ndg.records, expected)


  def testRandomize(self):
    ndg = NetworkDataGenerator()
    filename = os.path.join(self.dirName, "test_data/multi_sample.csv")
    ndg.split(filename, 3, False)

    random.seed(1)
    ndg.randomizeData()

    dataOutputFile = os.path.join(
        self.dirName, "test_data/multi_sample_split.csv")
    categoriesOutputFile = os.path.join(
        self.dirName, "test_data/multi_sample_categories.json")
    success = ndg.saveData(dataOutputFile, categoriesOutputFile)

    randomizedIDs = []
    dataTable = pandas.read_csv(dataOutputFile)
    for _, values in dataTable.iterrows():
      record = values.to_dict()
      idx = record["_sequenceID"]
      if idx.isdigit() and (not randomizedIDs or randomizedIDs[-1] != idx):
        randomizedIDs.append(idx)

    self.assertNotEqual(randomizedIDs, range(len(randomizedIDs)))


  def testSaveData(self):
    ndg = NetworkDataGenerator()
    filename = os.path.join(self.dirName, "test_data/multi_sample.csv")
    ndg.split(filename, 3, False)
    dataOutputFile = os.path.join(
        self.dirName, "test_data/multi_sample_split.csv")
    categoriesOutputFile = os.path.join(
        self.dirName, "test_data/multi_sample_categories.json")
    success = ndg.saveData(dataOutputFile, categoriesOutputFile)
    self.assertTrue(success)

    dataTable = pandas.read_csv(dataOutputFile).fillna("")

    types = {"_categories": "list",
             "_token": "string",
             "_sequenceID": "int",
             "_reset": "int",
             "ID": "string"}
    specials = {"_categories": "C",
                "_token": "",
                "_sequenceID": "S",
                "_reset": "R",
                "ID": ""}
    
    expected_records = [record for data in self.expected for record in data]
    expected_records.insert(0, specials)
    expected_records.insert(0, types)

    for idx, values in dataTable.iterrows():
      record = values.to_dict()
      if idx > 1:
        # csv values are strings, so cast the ints
        record["_sequenceID"] = int(record["_sequenceID"])
        record["_reset"] = int(record["_reset"])
      self.assertDictEqual(record, expected_records[idx])

    with open(categoriesOutputFile) as f:
      categories = json.load(f)

    expected_categories = {"kitchen": 0, "environment": 1, "not helpful": 2}
    self.assertDictEqual(categories, expected_categories)


  def testSaveDataIncorrectType(self):
    ndg = NetworkDataGenerator()
    filename = os.path.join(self.dirName, "test_data/multi_sample.csv")
    dataOutputFile = os.path.join(
        self.dirName, "test_data/multi_sample_split.csv")
    categoriesOutputFile = os.path.join(
        self.dirName, "test_data/multi_sample_categories.csv")
    ndg.split(filename, 3, False)

    with self.assertRaises(TypeError):
      ndg.saveData(dataOutputFile, categoriesOutputFile)


  def testFileRecordStreamReadData(self):
    ndg = NetworkDataGenerator()
    filename = os.path.join(self.dirName, "test_data/multi_sample.csv")
    ndg.split(filename, 3, False)
    dataOutputFile = os.path.join(
        self.dirName, "test_data/multi_sample_split.csv")
    categoriesOutputFile = os.path.join(
        self.dirName, "test_data/multi_sample_categories.json")
    ndg.saveData(dataOutputFile, categoriesOutputFile)

    # If no error is raised, then the data is in the correct format
    frs = FileRecordStream(dataOutputFile)


if __name__ == "__main__":
  unittest.main()
