#!/bin/bash

# Build NLP models from scratch. This assumes the following:
#   nupic.research repo is in the users nta/ directory
#   data is located in engine/data.csv

echo
echo "Creating and training NLP models, saving to engine/"

echo
echo "HTM_sensor_simple_tp_knn..."
python ~/nta/nupic.research/htmresearch/frameworks/nlp/imbu.py \
  --dataPath engine/data.csv \
  --cacheRoot cache \
  -c ~/nta/nupic.research/projects/nlp/data/network_configs/imbu_sensor_simple_tp_knn.json \
  -m HTM_sensor_simple_tp_knn \
  --savePath engine/HTM_sensor_simple_tp_knn \
  --noQueries

echo
echo "HTM_sensor_knn..."
python ~/nta/nupic.research/htmresearch/frameworks/nlp/imbu.py \
  --dataPath engine/data.csv \
  --cacheRoot cache \
  -c ~/nta/nupic.research/projects/nlp/data/network_configs/imbu_sensor_knn.json \
  -m HTM_sensor_knn \
  --savePath engine/HTM_sensor_knn \
  --noQueries

echo
echo "CioWordFingerprint..."
python ~/nta/nupic.research/htmresearch/frameworks/nlp/imbu.py \
  --dataPath engine/data.csv \
  --cacheRoot cache \
  -m CioWordFingerprint \
  --savePath engine/CioWordFingerprint \
  --noQueries

echo
echo "CioDocumentFingerprint..."
python ~/nta/nupic.research/htmresearch/frameworks/nlp/imbu.py \
  --dataPath engine/data.csv \
  --cacheRoot cache \
  -m CioDocumentFingerprint \
  --savePath engine/CioDocumentFingerprint \
  --noQueries

echo
echo "Keywords..."
python ~/nta/nupic.research/htmresearch/frameworks/nlp/imbu.py \
  --dataPath engine/data.csv \
  --cacheRoot cache \
  -m Keywords \
  --savePath engine/Keywords \
  --noQueries

echo "Done building models."
