# Parameters to generate the artificial sensor data
OUTFILE_NAME = 'white_noise'
SEQUENCE_LENGTH = 400
NUM_CATEGORIES = 3
NUM_RECORDS = 5  * SEQUENCE_LENGTH * NUM_CATEGORIES
WHITE_NOISE_AMPLITUDES = [0.0, 0.5]
SIGNAL_AMPLITUDES = [1.0]
SIGNAL_MEANS = [0.0, 10.0]
SIGNAL_PERIODS = [20.0]

# Additional parameters to run the classification experiments 
RESULTS_DIR = "results"
MODEL_PARAMS_DIR = 'model_params'
DATA_DIR = 'data'

# Verbosity of the trainNetwork phase
VERBOSITY = 0

# Whether to use a JSON config file of the config generator.
USE_JSON_CONFIG = True