#!/usr/bin/env python
# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2015, Numenta, Inc.  Unless you have an agreement
# with Numenta, Inc., for a separate license for this software code, the
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

from model_params.template_model_params import MODEL_PARAMS as TEMPLATE_MODEL_PARAMS
import csv
import os

from settings import SIGNAL_TYPES, MODEL_PARAMS_DIR, DATA_DIR


class IncorrectHeadersException(Exception):
  pass

def createModelParams(modelParamsDir, modelParamsName, fileName):
  
  # get the scalar values
  values = []
  with open(fileName, 'rU') as inputFile:
    csvReader = csv.reader(inputFile)
    headers = csvReader.next()
    
    # skip the rest of the header rows
    csvReader.next()
    csvReader.next()
    
    if headers[0] != 'x':
      raise IncorrectHeadersException("first column should be named 'x' but is '%s'" %headers[0])
    if headers[1] != 'y':
      raise IncorrectHeadersException("first column should be named 'y' but is '%s'" %headers[1])
  
    for line in csvReader:
      values.append(float(line[1]))
      
  # make sure the directory exists
  if not os.path.exists(modelParamsDir):
      os.makedirs(modelParamsDir)
      
  # make sure there is an init file so that we can import the model_params file later 
  with open("%s/%s" % (modelParamsDir, "__init__.py"), 'wb') as initFile:
    initFile.write("")
    
  # write the new model_params file
  with open("%s/%s.py" % (modelParamsDir, modelParamsName), 'wb') as modelParamsFile:
    minValue = min(values)
    maxValue = max(values)
    mp = TEMPLATE_MODEL_PARAMS
    mp['modelParams']['sensorParams']['encoders']['y']['maxval'] = maxValue
    mp['modelParams']['sensorParams']['encoders']['y']['minval'] = minValue
    modelParamsFile.write("MODEL_PARAMS = %s" % repr(mp))
    

def findMinMax(fileName):
  
  # get the scalar values
  values = []
  with open(fileName, 'rU') as inputFile:
    csvReader = csv.reader(inputFile)
    headers = csvReader.next()
    
    # skip the rest of the header rows
    csvReader.next()
    csvReader.next()
    
    if headers[0] != 'x':
      raise IncorrectHeadersException("first column should be named 'x' but is '%s'" %headers[0])
    if headers[1] != 'y':
      raise IncorrectHeadersException("first column should be named 'y' but is '%s'" %headers[1])
  
    for line in csvReader:
      values.append(float(line[1]))
      

  return min(values), max(values)
  


if __name__ == "__main__":
  
  for signal_type in SIGNAL_TYPES:
    inputFileName = '%s/%s.csv' % (DATA_DIR, signal_type)
    paramsName = '%s_model_params' % signal_type
    createModelParams(MODEL_PARAMS_DIR, paramsName, inputFileName)