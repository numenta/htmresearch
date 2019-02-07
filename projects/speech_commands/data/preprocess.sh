#!/usr/bin/env bash
for i in {0..24};
do
    python process_dataset.py -s speech_commands -d speech_commands/$i ;
done