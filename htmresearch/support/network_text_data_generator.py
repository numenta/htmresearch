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
"""
This file contains a class that tokenizes, randomizes, and writes the data to a
file in the format of the network API.
"""

import argparse
import csv
import os
import pprint
import random
import string

from collections import defaultdict, OrderedDict

from htmresearch.support.csv_helper import readCSV
from htmresearch.support.text_preprocess import TextPreprocess

try:
  import simplejson as json
except ImportError:
  import json



class NetworkDataGenerator(object):
  """Class for generating data in the format for a record stream."""


  def  __init__(self):
    """
    Column headers are marked "private" with a leading underscore in order to
    distingush them from dictinonary keys used in the Network API.

    Note: a reset marks the first item of a new sequence.
    """
    self.records = []
    self.fieldNames = ["_token", "_category", "_sequenceId", "_reset", "ID"]
    self.types = {"_token": "string",
                  "_category": "list",
                  "_sequenceId": "int",
                  "_reset": "int",
                  "ID": "string"}
    self.specials = {"_token": "",
                     "_category": "C",
                     "_sequenceId": "S",
                     "_reset": "R"}

    # len(self.categoryToId) gives each category a unique id w/o having
    # duplicates
    self.categoryToId = defaultdict(lambda: len(self.categoryToId))
    
    self.sequenceCount = 0


  def setupData(self, dataPath, numLabels=0, ordered=False, stripCats=False, seed=42, **kwargs):
    """
    Main method of this class. Use for setting up a network data file.
    
    @param dataPath        (str)    Path to CSV file.
    @param numLabels       (int)    Number of columns of category labels.
    @param textPreprocess  (bool)   True will preprocess text while tokenizing.
    @param ordered         (bool)   Keep data samples (sequences) in order,
                                    otherwise randomize.
    @param seed            (int)    Random seed.
    
    @return dataFileName   (str)    Network data file name; same directory as
                                    input data file.
    """
    self.split(dataPath, numLabels, **kwargs)
  
    if not ordered:
      self.randomizeData(seed)
    
    filename, ext = os.path.splitext(dataPath)
    classificationFileName = "{}_category.json".format(filename)
    dataFileName = "{}_network{}".format(filename, ext)
    
    if stripCats:
      self._stripCategories()
  
    self.saveData(dataFileName, classificationFileName)
    
    return dataFileName
  

  def split(self, filePath, numLabels, textPreprocess=False, seed=42,
            abbrCSV="", contrCSV="", ignoreCommon=100,
            removeStrings="[identifier deleted]", correctSpell=True):
    """
    Split all the comments in a file into tokens. Preprocess if necessary.
    
    @param filePath        (str)    Path to csv file
    @param numLabels       (int)    Number of columns of category labels.
    @param textPreprocess  (bool)   True will preprocess text while tokenizing.
    @param seed            (int)    Random seed.
    
    Please see TextPreprocess tokenize() for the other parameters; they're only
    used when textPreprocess is True.
    """
    dataDict = readCSV(filePath, numLabels=numLabels)
    if dataDict is None:
      raise Exception("Could not read CSV.")

    preprocessor = TextPreprocess(abbrCSV=abbrCSV, contrCSV=contrCSV)
    expandAbbr = (abbrCSV != "")
    expandContr = (contrCSV != "")

    for i, uniqueID in enumerate(dataDict.keys()):
      comment, categories = dataDict[uniqueID]
      # Convert the categories to a string of their IDs
      categories = string.join([str(self.categoryToId[c]) for c in categories])

      if textPreprocess:
        tokens = preprocessor.tokenize(
            comment, ignoreCommon, removeStrings, correctSpell, expandAbbr,
            expandContr)
      else:
        tokens = preprocessor.tokenize(comment)

      data = self._formatSequence(tokens, categories, i, uniqueID)

      self.records.append(data)
      self.sequenceCount += 1


  def _stripCategories(self):
    """Erases the categories, replacing them with the sequence number."""
    for data in self.records:
      for record in data:
        record["_category"] = record["_sequenceId"]


  @staticmethod
  def _formatSequence(tokens, categories, seqID, uniqueID):
    """Write the sequence of data records for this sample."""
    record = {"_category":categories,
              "_sequenceId":seqID}
    data = []
    reset = 1
    for t in tokens:
      tokenRecord = record.copy()
      tokenRecord["_token"] = t
      tokenRecord["_reset"] = reset
      tokenRecord["ID"] = uniqueID
      reset = 0
      data.append(tokenRecord)

    return data
  

  def randomizeData(self, seed=42):
    random.seed(seed)
    random.shuffle(self.records)


  def saveData(self, dataOutputFile, categoriesOutputFile):
    """
    Save the processed data and the associated category mapping.
    @param dataOutputFile       (str)   Location to save data
    @param categoriesOutputFile (str)   Location to save category map
    @return                     (str)   Path to the saved data file iff
                                        saveData() is successful.
    """
    if self.records is None:
      return False

    if not dataOutputFile.endswith("csv"):
      raise TypeError("data output file must be csv.")
    if not categoriesOutputFile.endswith("json"):
      raise TypeError("category output file must be json")

    # Ensure directory exists
    dataOutputDirectory = os.path.dirname(dataOutputFile)
    if not os.path.exists(dataOutputDirectory):
      os.makedirs(dataOutputDirectory)

    categoriesOutputDirectory = os.path.dirname(categoriesOutputFile)
    if not os.path.exists(categoriesOutputDirectory):
      os.makedirs(categoriesOutputDirectory)

    with open(dataOutputFile, "w") as f:
      # Header
      writer = csv.DictWriter(f, fieldnames=self.fieldNames)
      writer.writeheader()

      # Types
      writer.writerow(self.types)

      # Special characters
      writer.writerow(self.specials)

      for data in self.records:
        for record in data:
          writer.writerow(record)

    with open(categoriesOutputFile, "w") as f:
      f.write(json.dumps(self.categoryToId,
                         sort_keys=True,
                         indent=4,
                         separators=(",", ": ")))

    return dataOutputFile


  def generateSequence(self, text, preprocess=False):
    """
    Return a list of lists representing the text sequence in network data 
    format. Does not preprocess the text.
    """
    # TODO: enable text preprocessing; abstract out the logic in split() into a common method.
    tokens = TextPreprocess().tokenize(text)
    cat = [-1]
    self.sequenceCount += 1
    uniqueID = "q"
    data = self._formatSequence(tokens, cat, self.sequenceCount, uniqueID)

    return data


  def reset(self):
    self.records = []
    self.fieldNames = ["token", "_sequenceId", "_reset", "ID"]
    self.types = {"token": "string",
                  "_sequenceId": "int",
                  "_reset": "int",
                  "ID": "string"}
    self.specials = {"token": "",
                     "_sequenceId": "S",
                     "_reset": "R"}

    self.categoryToId.clear()


  @staticmethod
  def getSamples(netDataFile):
    """
    Returns samples joined at reset points.
    @param netDataFile  (str)         Path to file (in the FileRecordStream
                                      format).
    @return samples     (OrderedDict) Keys are sample number (in order they are
                                      read in). Values are two-tuples of sample
                                      text and category ints.
    """
    try:
      with open(netDataFile) as f:
        reader = csv.reader(f)
        header = next(reader, None)
        next(reader, None)
        resetIdx = next(reader).index("R")
        tokenIdx = header.index("_token")
        catIdx = header.index("_category")
        idIdx = header.index("ID")

        currentSample = []
        samples = OrderedDict()
        for line in reader:
          if int(line[resetIdx]) == 1:
            if len(currentSample) != 0:
              samples[line[idIdx]] = ([" ".join(currentSample)],
                                      [int(c) for c in line[catIdx].split(" ")])
            currentSample = [line[tokenIdx]]
          else:
            currentSample.append(line[tokenIdx])
        samples[line[idIdx]] = ([" ".join(currentSample)],
                                [int(c) for c in line[catIdx].split(" ")])

        return samples

    except IOError as e:
      print "Could not open the file {}.".format(netDataFile)
      raise e


  @staticmethod
  def getClassifications(networkDataFile):
    """
    Returns the classifications at the indices where the data sequences
    reset.
    @param networkDataFile  (str)     Path to file in the FileRecordStream
                                      format
    @return                 (list)    list of string versions of the
                                      classifications
    Sample output: ["0 1", "1", "1 2 3"]
    """
    try:
      with open(networkDataFile) as f:
        reader = csv.reader(f)
        next(reader, None)
        next(reader, None)
        specials = next(reader)
        resetIdx = specials.index("R")
        classIdx = specials.index("C")

        classifications = []
        for line in reader:
          if int(line[resetIdx]) == 1:
            classifications.append(line[classIdx])
        return classifications

    except IOError as e:
      print "Could not open the file {}.".format(networkDataFile)
      raise e


  @staticmethod
  def getNumberOfTokens(networkDataFile):
    """
    Returns the number of tokens for each sequence
    @param networkDataFile  (str)     Path to file in the FileRecordStream
                                      format
    @return                 (list)    list of number of tokens
    """
    try:
      with open(networkDataFile) as f:
        reader = csv.reader(f)
        next(reader, None)
        next(reader, None)
        resetIdx = next(reader).index("R")

        count = 0
        numTokens = []
        for line in reader:
          if int(line[resetIdx]) == 1:
            if count != 0:
              numTokens.append(count)
            count = 1
          else:
            count += 1
        numTokens.append(count)
        return numTokens

    except IOError as e:
      print "Could not open the file {}.".format(networkDataFile)
      raise e


  @staticmethod
  def getResetsIndices(networkDataFile):
    """Returns the indices at which the data sequences reset."""
    try:
      with open(networkDataFile) as f:
        reader = csv.reader(f)
        next(reader, None)
        next(reader, None)
        resetIdx = next(reader).index("R")

        resets = []
        for i, line in enumerate(reader):
          if int(line[resetIdx]) == 1:
            resets.append(i)
        return resets

    except IOError as e:
      print "Could not open the file {}.".format(networkDataFile)
      raise e



if __name__ == "__main__":

  parser = argparse.ArgumentParser(description="Create data for network API")

  parser.add_argument("-fin", "--filename",
                      type=str,
                      required=True,
                      help="path to input file. REQUIRED")
  parser.add_argument("-fo", "--dataOutputFile",
                      default="network_experiment/data.csv",
                      type=str,
                      help="File to write processed data to.")
  parser.add_argument("-c", "--categoriesOutputFile",
                      type=str,
                      default="network_experiment/categories.json",
                      help="File to write the categories to ID mapping.")
  parser.add_argument("--numLabels",
                      type=int,
                      default=3,
                      help="Column number(s) of the category label.")
  parser.add_argument("--textPreprocess",
                      action="store_true",
                      default=False,
                      help="Basic preprocessing. Use specific tags for custom "
                      "preprocessing")
  parser.add_argument("--ignoreCommon",
                      default=100,
                      type=int,
                      help="Number of common words to ignore")
  parser.add_argument("--removeStrings",
                      type=str,
                      default=["[identifier deleted]"],
                      nargs="+",
                      help="Strings to remove in preprocessing")
  parser.add_argument("--correctSpell",
                      default=True,
                      action="store_false",
                      help="Whether or not to use spelling correction")
  parser.add_argument("--abbrCSV",
                      default="",
                      help="Path to CSV of abbreviations data")
  parser.add_argument("--contrCSV",
                      default="",
                      help="Path to CSV of contractions data")
  parser.add_argument("--randomize",
                      default=False,
                      action="store_true",
                      help="Whether or not to randomize the order of the data "
                      "samples before saving")

  options = vars(parser.parse_args())

  pprint.pprint(options)
  print ("Note: preprocessing params only take affect if textPreprocess "
         "argument is set.")

  dataGenerator = NetworkDataGenerator()
  dataGenerator.split(**options)

  if options["randomize"]:
    dataGenerator.randomizeData()

  outFile = dataGenerator.saveData(**options)
