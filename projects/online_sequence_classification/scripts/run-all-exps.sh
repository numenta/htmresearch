#!/usr/bin/env bash

target_dir=$1
./generate-plots.sh $target_dir rawAS_tmPredictedActiveCells.py
./generate-plots.sh $target_dir rawAS_tmActiveCells.py
./generate-plots.sh $target_dir rollingAS_tmActiveCells.py