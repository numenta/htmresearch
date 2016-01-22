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
This file contains CSV utility functions to use with nupic.fluent experiments.
"""

import csv
import itertools
import numpy
import os

from collections import defaultdict, OrderedDict


def readCSV(csvFile, numLabels=0):
  """
  Read in a CSV file w/ the following formatting:
    - one header row
    - one page
    - one column of samples, followed by column(s) of labels

  @param csvFile         (str)          File name for the input CSV.
  @param numLabels       (int)          Number of columns of category labels.
  @return                (dict)         Keys are sample IDs, values are 3-tuples
                                        of sample (str), categories (list of
                                        str), sample number (int).
  """
  try:
    with open(csvFile, "rU") as f:
      reader = csv.reader(f)
      headers = next(reader, None)
      try:
        sampleIdx = headers.index("Sample")
        idIdx = headers.index("ID")
      except ValueError as e:
        print ("Could not find 'ID' and/or 'Sample' columns, so assuming "
               "they are 0 and 2, respectively.")
        sampleIdx = 2
        idIdx = 0
      
      dataDict = {}

      if numLabels > 0:
        labelIdx = range(sampleIdx + 1, sampleIdx + 1 + numLabels)
        for lineNumber, line in enumerate(reader):
          dataDict[lineNumber] = (line[sampleIdx],
                                  [line[i] for i in labelIdx if line[i]],
                                  line[idIdx])
      else:
        for lineNumber, line in enumerate(reader):
          dataDict[lineNumber] = (line[sampleIdx], [], line[idIdx])

      return dataDict

  except IOError as e:
    print e


def mapLabelRefs(dataDict):
  """
  Replace the label strings in dataDict with corresponding ints.

  @return (tuple)   (ordered list of category names, dataDict with names
                    replaced by array of category indices)
  """
  labelRefs = [label for label in set(
    itertools.chain.from_iterable([x[1] for x in dataDict.values()]))]

  for recordNumber, data in dataDict.iteritems():
    dataDict[recordNumber] = (data[0], numpy.array(
      [labelRefs.index(label) for label in data[1]]), data[2])

  return labelRefs, dataDict


def bucketCSVs(csvFile, bucketIdx=2):
  """Write the individual buckets in csvFile to their own CSV files."""
  try:
    with open(csvFile, "rU") as f:
      reader = csv.reader(f)
      headers = next(reader, None)
      dataDict = OrderedDict()
      for lineNumber, line in enumerate(reader):
        if line[bucketIdx] in dataDict:
          dataDict[line[bucketIdx]].append(line)
        else:
          # new bucket
          dataDict[line[bucketIdx]] = [line]
  except IOError as e:
    print e

  filePaths = []
  for i, (_, lines) in enumerate(dataDict.iteritems()):
    bucketFile = csvFile.replace(".", "_"+str(i)+".")
    writeCSV(lines, headers, bucketFile)
    filePaths.append(bucketFile)

  return filePaths


def readDir(dirPath, numLabels, modify=False):
  """
  Reads in data from a directory of CSV files; assumes the directory only
  contains CSV files.

  @param dirPath            (str)          Path to the directory.
  @param numLabels          (int)          Number of columns of category labels.
  @param modify             (bool)         Map the unix friendly category names
                                           to the actual names. 0 -> /, _ -> " "

  @return samplesDict       (defaultdict)  Keys are CSV names, values are
      OrderedDicts, where the keys/values are as specified in readCSV().
  """
  samplesDict = defaultdict(list)
  for _, _, files in os.walk(dirPath):
    for f in files:
      basename, extension = os.path.splitext(os.path.basename(f))
      if "." in basename and extension == ".csv":
        category = basename.split(".")[-1]
        if modify:
          category = category.replace("0", "/")
          category = category.replace("_", " ")
        samplesDict[category] = readCSV(
          os.path.join(dirPath, f), numLabels=numLabels)

  return samplesDict


def writeCSV(data, headers, csvFile):
  """Write data with column headers to a CSV."""
  with open(csvFile, "wb") as f:
    writer = csv.writer(f, delimiter=",")
    writer.writerow(headers)
    writer.writerows(data)


def writeFromDict(dataDict, headers, csvFile):
  """
  Write dictionary to a CSV, where keys are row numbers and values are a list.
  """
  with open(csvFile, "wb") as f:
    writer = csv.writer(f, delimiter=",")
    writer.writerow(headers)
    for row in sorted(dataDict.keys()):
      writer.writerow(dataDict[row])


def readDataAndReshuffle(args, categoriesInOrderOfInterest=None):
  """
  Read data file specified in args, optionally reshuffle categories, print out
  some statistics, and return various data structures. This routine is pretty
  specific and only used in some simple test scripts.

  categoriesInOrderOfInterest (list) Optional list of integers representing
                                     the priority order of various categories.
                                     The categories in the original data file
                                     will be reshuffled to the order in this
                                     array, up to args.numLabels.

  Returns the tuple:
    (dataset, labelRefs, documentCategoryMap, documentTextMap)

  Return format:
      dataset = [
        ["fox eats carrots", [0], docId],
        ["fox eats peppers", [0], docId],
        ["carrots are healthy", [1], docId],
        ["peppers is healthy", [1], docId],
      ]

      labelRefs = [Category0Name, Category1Name, ...]

      documentCategoryMap = {
        docId: [categoryIndex0, categoryIndex1, ...],
        docId: [categoryIndex0, categoryIndex1, ...],
                :
      }

      documentTextMap = {
        docId: documentText,
        docId: documentText,
                :
      }

  """
  # Read data
  dataDict = readCSV(args.dataPath, 1)
  labelRefs, dataDict = mapLabelRefs(dataDict)
  if categoriesInOrderOfInterest is None:
      categoriesInOrderOfInterest = range(0,args.numLabels)
  else:
    categoriesInOrderOfInterest=categoriesInOrderOfInterest[0:args.numLabels]

  # Select data based on categories of interest. Shift category indices down
  # so we go from 0 to numLabels-1
  dataSet = []
  documentTextMap = {}
  counts = numpy.zeros(len(labelRefs))
  for document in dataDict.itervalues():
    try:
      docId = int(document[2])
    except:
      raise RuntimeError("docId "+str(docId)+" is not an integer")
    oldCategoryIndex = document[1][0]
    documentTextMap[docId] = document[0]
    if oldCategoryIndex in categoriesInOrderOfInterest:
      newIndex = categoriesInOrderOfInterest.index(oldCategoryIndex)
      dataSet.append([document[0], [newIndex], docId])
      counts[newIndex] += 1

  # For each document, figure out which categories it belongs to
  # Include the shifted category index
  documentCategoryMap = {}
  for doc in dataDict.iteritems():
    docId = int(doc[1][2])
    oldCategoryIndex = doc[1][1][0]
    if oldCategoryIndex in categoriesInOrderOfInterest:
      newIndex = categoriesInOrderOfInterest.index(oldCategoryIndex)
      v = documentCategoryMap.get(docId, [])
      v.append(newIndex)
      documentCategoryMap[docId] = v

  labelRefs = [labelRefs[i] for i in categoriesInOrderOfInterest]
  print "Total number of unique documents",len(documentCategoryMap)
  print "Category counts: ",counts
  print "Categories in training/test data:", labelRefs

  return dataSet, labelRefs, documentCategoryMap, documentTextMap


