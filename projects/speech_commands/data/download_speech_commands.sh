#!/usr/bin/env sh
set -e

FILE_NAME=speech_commands_v0.01.tar.gz
URL=http://download.tensorflow.org/data/$FILE_NAME
echo "downloading $URL...\n"
wget $URL

DATASET_FOLDER=speech_commands_test
echo "extracting $FILE_NAME..."
mkdir -p $DATASET_FOLDER
tar -xzf $FILE_NAME -C $DATASET_FOLDER

echo "splitting the dataset into train, validation and test sets..."
python split_dataset.py --root=$DATASET_FOLDER --size=small

echo "done"
