spParamSmall = {
  "inputDimensions": (1024, 1),
  "columnDimensions": (512, 1),
  "potentialRadius": 1024,
  "globalInhibition": True,
  "numActiveColumnsPerInhArea": 10,
  "stimulusThreshold": 1,
  "synPermInactiveDec": 0.001,
  "synPermActiveInc": 0.001,
  "synPermConnected": 0.1,
  "minPctOverlapDutyCycle": 0.0,
  "minPctActiveDutyCycle": 0.0,
  "dutyCyclePeriod": 1000,
  "maxBoost": 2.0,
  "seed": 1936
}


spParamNoBoosting = {
  "inputDimensions": (1024, 1),
  "columnDimensions": (1024, 1),
  "potentialRadius": 1024,
  "potentialPct": 0.5,
  "globalInhibition": True,
  "numActiveColumnsPerInhArea": 20,
  "stimulusThreshold": 1,
  "synPermInactiveDec": 0.01,
  "synPermActiveInc": 0.02,
  "synPermConnected": 0.5,
  "minPctOverlapDutyCycle": 0.0,
  "minPctActiveDutyCycle": 0.0,
  "dutyCyclePeriod": 1000,
  "maxBoost": 1.0,
  "seed": 1936
}


spParamWithBoosting = {
  "inputDimensions": (1024, 1),
  "columnDimensions": (1024, 1),
  "potentialRadius": 1024,
  "potentialPct": 1.0,
  "globalInhibition": True,
  "numActiveColumnsPerInhArea": 20,
  "stimulusThreshold": 1,
  "synPermInactiveDec": 0.01,
  "synPermActiveInc": 0.02,
  "synPermConnected": 0.5,
  "minPctOverlapDutyCycle": 0.0,
  "minPctActiveDutyCycle": 0.001,
  "dutyCyclePeriod": 1000,
  "maxBoost": 5.0,
  "seed": 1936
}


spParamTopologyWithBoosting = {
  "inputDimensions": (32, 32),
  "columnDimensions": (32, 32),
  "potentialRadius": 5,
  "potentialPct": .8,
  "globalInhibition": False,
  "numActiveColumnsPerInhArea": -1,
  "localAreaDensity": .02,
  "wrapAround": False,
  "stimulusThreshold": 1,
  "synPermInactiveDec": 0.01,
  "synPermActiveInc": 0.02,
  "synPermConnected": 0.5,
  "minPctOverlapDutyCycle": 0.0,
  "minPctActiveDutyCycle": 0.001,
  "dutyCyclePeriod": 1000,
  "maxBoost": 1.0,
  "seed": 1936
}