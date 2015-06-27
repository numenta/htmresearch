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



if __name__ == "__main__":
  
  for signal_type in SIGNAL_TYPES:
    inputFileName = '%s/%s.csv' % (DATA_DIR, signal_type)
    paramsName = '%s_model_params' % signal_type
    createModelParams(MODEL_PARAMS_DIR, paramsName, inputFileName)