# Signal types can be: 'binary', 'sine', 'triangle'
SIGNAL_TYPES = [
  'binary',
  'sine', 
  'triangle'
]

# Parameters to generate the artificial sensor data
DATA_DIR = 'data'
NUM_CATEGORIES = [2]
WHITE_NOISE_AMPLITUDES = [0.0]
SIGNAL_AMPLITUDES = [10.0]
SIGNAL_MEANS = [0.0]

# Number of phases. Eg: Train (1) SP, (2) TM, (3) TP, (4) Classifier, (5) Test
NUM_PHASES = [5]

# Number of time each phase repeats
NUM_REPS = [10]

# Verbosity of network
VERBOSITY = 0

# Whether to use a JSON config file of the config generator.
# See network_configs.json for an example of standard config file.
USE_JSON_CONFIG = False
