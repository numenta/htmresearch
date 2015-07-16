import csv
import json
import fileinput
import math

from collections import defaultdict
import settings
import os

from settings import METRICS, SENSORS, INPUT_DATA_FILE, CONVERTED_DATA_DIR, MODEL_PARAMS_DIR, INPUT_DATA_DIR
from prototype.model_params_template import MODEL_PARAMS as TEMPLATE_MODEL_PARAMS

DIRS = [CONVERTED_DATA_DIR, MODEL_PARAMS_DIR, INPUT_DATA_DIR]
for directory in DIRS:
  if not os.path.exists(directory):
      os.makedirs(directory)

def convertMag(raw):
  """
  mag raw[0..5]
  -mx = raw[0]+256*raw[1]-65536*(raw[1]>127));
  -my = raw[2]+256*raw[3]-65536*(raw[3]>127));
  mz = raw[4]+256*raw[5]-65536*(raw[5]>127));
  """

  mx = -(raw[0] + 256 * raw[1] - 65536 * (raw[1] > 127))
  my = -(raw[2] + 256 * raw[3] - 65536 * (raw[3] > 127))
  mz = raw[4] + 256 * raw[5] - 65536 * (raw[5] > 127)
  return mx, my, mz


def convertGyro(raw):
  """
  gyro raw[0..5]
  gx = raw[2]+256*raw[3]-65536*(raw[3]>127));
  gy = raw[0]+256*raw[1]-65536*(raw[1]>127));
  gz = raw[4]+256*raw[5]-65536*(raw[5]>127));
  """

  gx = raw[2] + 256 * raw[3] - 65536 * (raw[3] > 127)
  gy = raw[0] + 256 * raw[1] - 65536 * (raw[1] > 127)
  gz = raw[4] + 256 * raw[5] - 65536 * (raw[5] > 127)
  return gx, gy, gz


def convertAccel(raw):
  """
  accel raw[0..2]
  ax = raw[0]-256*(raw[0]>127));
  -ay = raw[1]-256*(raw[1]>127));
  az = raw[2]-256*(raw[2]>127));
  """

  ax = raw[0] - 256 * (raw[0] > 127)
  ay = -(raw[1] - 256 * (raw[1] > 127))
  az = raw[2] - 256 * (raw[2] > 127)

  return ax, ay, az


def convert(reading, sensor):
  """
  Convert reading according to the sensor type.
  """

  if sensor == "mag":
    x, y, z = convertMag(reading['raw'])
  elif sensor == "accel":
    x, y, z = convertAccel(reading['raw'])
  elif sensor == "gyro":
    x, y, z = convertGyro(reading['raw'])

  return x, y, z


def filterBySensorTypeAndConvert(rawAggregatedReadings):
  for patientId in rawAggregatedReadings:

    # find rotation parameters
    accelReadings = [reading for reading in rawAggregatedReadings[patientId] if reading['sensor'] == 'accel']
    print "PatientID: %s" %patientId
    phi, omega = findRotationParams(accelReadings)


    # filter readings by sensor type 
    for sensor in SENSORS:
      readings = [reading for reading in rawAggregatedReadings[patientId] if reading['sensor'] == sensor]

      # get min / max of each metric
      values = {metric: [] for metric in METRICS}
      convertedReadings = []
      for reading in readings:
        x_converted, y_converted, z_converted = convert(reading, sensor)
        x, y, z = rotate(x_converted, y_converted, z_converted, phi, omega)
        #print "before rotation (x,y,z) = (%s,%s,%s)" %(x_converted, y_converted, z_converted)
        #print "after rotation (x,y,z) = (%s,%s,%s)" %(x,y,z)
        #print "" 
          
        values['x'].append(x)
        values['y'].append(y)
        values['z'].append(z)
        convertedReadings.append((x, y, z))

      create_model_params(patientId, sensor, values)

      for metric in METRICS:
        with open('%s/%s_%s_%s.csv' % (CONVERTED_DATA_DIR, metric, sensor, patientId), "wb") as outputFile:
          csvWriter = csv.writer(outputFile)
          csvWriter.writerow(["metric_value"])
          for value in values[metric]:
            csvWriter.writerow([value])


def create_model_params(patient, sensor, values):
  with open("%s/%s" % (MODEL_PARAMS_DIR, "__init__.py"), 'wb') as initFile:
    initFile.write("")
    
  for metric in METRICS:
    with open("%s/%s_%s_%s.py" % (MODEL_PARAMS_DIR, metric, sensor, patient), 'wb') as modelParamsFile:
      min_value = min(values[metric])
      max_value = max(values[metric])
      mp = TEMPLATE_MODEL_PARAMS
      mp['modelParams']['sensorParams']['encoders']['_classifierInput']['maxval'] = max_value
      mp['modelParams']['sensorParams']['encoders']['_classifierInput']['minval'] = min_value
      mp['modelParams']['sensorParams']['encoders']['metric_value']['maxval'] = max_value
      mp['modelParams']['sensorParams']['encoders']['metric_value']['minval'] = min_value
      modelParamsFile.write("MODEL_PARAMS = %s" % repr(mp))


def writeReadings(readings, metric_name, baseFileName):
  outFileName = metric_name + '_' + baseFileName

  print "Writing %s with %s readings" % (outFileName, len(readings))
  with open('%s/%s' % (CONVERTED_DATA_DIR, outFileName), "wb") as outputFile:
    csvWriter = csv.writer(outputFile)
    csvWriter.writerow(["metric_value"])

    for reading in readings:
      csvWriter.writerow(reading)


def convertReadings(fileName):
  """
  Convert readings:
  - Aggregate raw readings by patientID,
  - Filter by sensor type,
  - Convert values according to sensor type,
  - Write to file.

  """

  aggregatedReadings = defaultdict(list)
  patient_ids = []  # keep track of all IDs seen so far

  with open(fileName, "rb") as inputFile:

    csvReader = csv.reader(inputFile)
    headers = csvReader.next()

    for line in csvReader:

      row = {}
      for i, x in enumerate(line):
        row[headers[i]] = x

      patient_id = row['password']
      patient_ids.append(patient_id)
      readings = json.loads(row['readings'])
      aggregatedReadings[patient_id].extend(readings)
      
    filterBySensorTypeAndConvert(aggregatedReadings)
    
    update_settings_file(list(set(patient_ids)))


def update_settings_file(patient_ids):
  """
  Update settings file with correct list of patient IDs
  """

  for line in fileinput.FileInput(settings.__file__,inplace=True):
    if "PATIENT_IDS" in line:
      print "PATIENT_IDS = %s".rstrip() % patient_ids
    else:
      print line.rstrip()
      

def findRotationParams(accelReadings):
    """
    Compute PHI=ATAN(-X/Y)
    Compute OMEGA=ATAN(Z/(X*SIN(PHI)+Y*COS(PHI)))
    """

    maxG2 = 256*1.20
    minG2 = 256*.8
    
    # max number of readings from the accelerometer sensor that will be used to find the values of Phi and Omega 
    maxAccelReadings = 25       
    firstUnstableReading = maxAccelReadings
    stable_values = defaultdict(list)
    for reading in accelReadings[:maxAccelReadings]:
        raw = reading['raw']
        x, y, z = convertAccel(raw)
        
        # magnitude of the acceleration force
        G2 = x**2+ y**2 + z**2
        
        # Find the last i for which G2 within an 5% band around 256
        if G2 > maxG2 or G2 < minG2:
          firstUnstableReading = accelReadings.index(reading)
          break
        else:
          stable_values['x'].append(x)
          stable_values['y'].append(y)
          stable_values['z'].append(z)
          
          
    if firstUnstableReading > 10:
      # Otherwise compute the average of the accelerometer x, y and z for the i initial data points. 
      # Let these averages be referred to as X, Y and Z.
      #
      # Compute phi=ATAN(-X/Y)
      # Compute omega=ATAN(Z/(X*SIN(phi)+Y*COS(phi)))
      
      X = mean(stable_values['x'])
      Y = mean(stable_values['y'])
      Z = mean(stable_values['z'])
      
      phi = math.atan(-X/Y)
      omega = math.atan(Z/(X*math.sin(phi) + Y*math.cos(phi)))
      print "Stable data: phi = %s, omega = %s" % (phi, omega)
    
      # just checking it's nicely rotated
      X1, Y1, Z1 = rotate(X,Y,Z,phi,omega)
      print "initial values: (X, Y, Z) = (%s, %s, %s)" % (X,Y,Z)
      print "rotated values: (X1,Y1,Z1) = (%s,%s,%s)" % (X1,Y1,Z1)
      print ""
    
    else:
      print "minG2, maxG2, G2 = %s, %s, %s" %(minG2, maxG2, G2)
      raise _UnstableDatasetError("Unstable data: not enough values of G2 (x^2 + y^2 + z^2) withing a 5% range of 256.")
      
    return phi, omega
 
def mean(array):
  return float(sum(array))/float(len(array))
   
def rotate(x, y, z, phi, omega):
  """
  X1=X*COS(PHI)+Y*SIN(PHI)
  Y1=-X*SIN(PHI)*COS(OM)+Y*COS(PHI)*COS(OM)+Z*SIN(OM)
  Z1=-X*SIN(PHI)*SIN(OM)-Y*COS(PHI)*SIN(OM)+Z*COS(OM)
  """
    
  x1 = x * math.cos(phi) + y * math.sin(phi)
  y1 = -x * math.sin(phi) * math.cos(omega) + y * math.cos(phi) * math.cos(omega) + z * math.sin(omega)
  z1 = -x * math.sin(phi) * math.sin(omega) -y * math.cos(phi) * math.sin(omega) + z * math.cos(omega)
  
  return x1, y1, z1
  
class _UnstableDatasetError(Exception):
  pass

if __name__ == '__main__':
  inputFile = "%s/%s" %(INPUT_DATA_DIR, INPUT_DATA_FILE)
  convertReadings(inputFile)

      
  
      
    