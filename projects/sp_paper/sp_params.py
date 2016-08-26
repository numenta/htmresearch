columnNumber = 2048

spParamNoBoosting = {
  "inputDimensions": (1024, 1),
  "columnDimensions": (columnNumber, 1),
  "potentialRadius": 1024,
  "globalInhibition": True,
  "numActiveColumnsPerInhArea": int(0.02 * columnNumber),
  "stimulusThreshold": 3,
  "synPermInactiveDec": 0.001,
  "synPermActiveInc": 0.001,
  "synPermConnected": 0.1,
  "minPctOverlapDutyCycle": 0.0,
  "minPctActiveDutyCycle": 0.0,
  "dutyCyclePeriod": 1000,
  "maxBoost": 2.0,
  "seed": 1936
}


spParamWithBoosting = {
  "inputDimensions": (1024, 1),
  "columnDimensions": (columnNumber, 1),
  "potentialRadius": 1024,
  "globalInhibition": True,
  "numActiveColumnsPerInhArea": int(0.02 * columnNumber),
  "stimulusThreshold": 3,
  "synPermInactiveDec": 0.001,
  "synPermActiveInc": 0.001,
  "synPermConnected": 0.1,
  "minPctOverlapDutyCycle": 0.0,
  "minPctActiveDutyCycle": 0.0,
  "dutyCyclePeriod": 1000,
  "maxBoost": 2.0,
  "seed": 1936
}