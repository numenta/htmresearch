#!/bin/bash

# Build NLP models from scratch. This assumes the following:
#   nupic.research repo is in the users nta/ directory
#   data is located in engine/data.csv

echo
echo "Creating and training NLP models, saving to engine/"

echo "  HTM_sensor_simple_tp_knn..."
python ~/nta/nupic.research/htmresearch/frameworks/nlp/imbu.py \
  --dataPath engine/sample_reviews/data.csv \
  -c ~/nta/nupic.research/projects/nlp/data/network_configs/imbu_sensor_simple_tp_knn.json \
  -m HTM_sensor_simple_tp_knn \
  --savePath engine/sample_reviews/HTM_sensor_simple_tp_knn \
  --noQueries

echo "  HTM_sensor_knn..."
python ~/nta/nupic.research/htmresearch/frameworks/nlp/imbu.py \
  --dataPath engine/sample_reviews/data.csv \
  -c ~/nta/nupic.research/projects/nlp/data/network_configs/imbu_sensor_knn.json \
  -m HTM_sensor_knn \
  --savePath engine/sample_reviews/HTM_sensor_knn \
  --noQueries

echo "  HTM_sensor_tm_simple_tp_knn..."
python ~/nta/nupic.research/htmresearch/frameworks/nlp/imbu.py \
  --dataPath engine/sample_reviews/data.csv \
  -c ~/nta/nupic.research/projects/nlp/data/network_configs/imbu_sensor_tm_simple_tp_knn.json \
  -m HTM_sensor_tm_knn \
  --savePath engine/sample_reviews/HTM_sensor_tm_knn \
  --noQueries

echo "  CioWordFingerprint..."
python ~/nta/nupic.research/htmresearch/frameworks/nlp/imbu.py \
  --dataPath engine/sample_reviews/data.csv \
  -m CioWordFingerprint \
  --savePath engine/sample_reviews/CioWordFingerprint \
  --noQueries

echo "  CioDocumentFingerprint..."
python ~/nta/nupic.research/htmresearch/frameworks/nlp/imbu.py \
  --dataPath engine/sample_reviews/data.csv \
  -m CioDocumentFingerprint \
  --savePath engine/sample_reviews/CioDocumentFingerprint \
  --noQueries

echo "  Keywords..."
python ~/nta/nupic.research/htmresearch/frameworks/nlp/imbu.py \
  --dataPath engine/sample_reviews/data.csv \
  -m Keywords \
  --savePath engine/sample_reviews/Keywords \
  --noQueries

echo "Done building models."
