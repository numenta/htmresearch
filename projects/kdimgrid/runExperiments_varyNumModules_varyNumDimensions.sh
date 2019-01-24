#!/bin/bash

set -e -x

cd "$(dirname "$0")"
pwd

DATA_DIR=$(python -c "import time; import uuid; print '{}-{}'.format(time.strftime('%Y%m%d-%H%M%S'), uuid.uuid1())")
mkdir -p ./data/$DATA_DIR

python -u generate_data.py $DATA_DIR --m 1 2 3 4 5 6 7 8 9 --k 3 4 5 6 --numTrials 30
python -u measure_unique_sidelength.py $DATA_DIR
