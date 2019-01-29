#!/bin/bash

set -e -x

cd "$(dirname "$0")"
pwd

DATA_DIR=data/1D_benchmark_filtered/$(python -c "import time; import uuid; print '{}-{}'.format(time.strftime('%Y%m%d-%H%M%S'), uuid.uuid1())")
mkdir -p ./data/$DATA_DIR

python -u generate_filtered_data.py $DATA_DIR --phaseResolution 0.2 --m 1 2 3 --k 1  --numTrials 2000
python -u measure_unique_sidelength_filtered.py $DATA_DIR
