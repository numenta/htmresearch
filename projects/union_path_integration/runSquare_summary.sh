#!/bin/bash

set -e -x

cd "$(dirname "$0")"
pwd

python run_summary.py --bumpType square --resultName results/convergenceSummary_square.json --repeat 1

python plot_summary.py --inFile results/convergenceSummary_square.json --outFile summary_square.pdf --squeezeLegend
