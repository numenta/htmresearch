#!/bin/bash

set -e -x

cd "$(dirname "$0")"
pwd

DATA_DIR=data/phr_k/$(python -c "import time; import uuid; print '{}-{}'.format(time.strftime('%Y%m%d-%H%M%S'), uuid.uuid1())")
mkdir -p ./data/$DATA_DIR

python -u generate_data.py $DATA_DIR --phaseResolution 0.4 0.2 0.1 0.05 0.025 --m 5 --k 3 4 5 6 --numTrials 100
python -u measure_unique_sidelength.py $DATA_DIR
