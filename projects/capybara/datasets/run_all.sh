#!/usr/bin/env bash

# Params
uci_labels='[0,1,2,3,4,5]'
uci_samples=2000
uci_chunk_size=200

# Get UCI data
python uci_har/download_dataset.py --output_dir=uci_har
python uci_har/convert_to_csv.py --input_dir=uci_har --output_dir=uci_har \
    --labels=${uci_labels} --nb_samples=${uci_samples}


# Convert UCI data
python convert_to_sequences.py -i uci_har/inertial_signals_train.csv \
    -o uci_sequences/inertial_signals -c ${uci_chunk_size}
python convert_to_sequences.py -i uci_har/inertial_signals_test.csv \
    -o uci_sequences/inertial_signals -c ${uci_chunk_size}
python convert_to_sequences.py -i uci_har/debug_train.csv \
    -o uci_sequences/debug -c ${uci_chunk_size}
python convert_to_sequences.py -i uci_har/debug_test.csv \
    -o uci_sequences/debug -c ${uci_chunk_size}

# Motifs dataset
pushd SyntheticData
python generate_synthetic_data.py
popd

# Generate artificial data
pushd artificial
python generate_artificial_data.py
popd

# Convert artificial data
python convert_to_sequences.py \
    -i artificial/binary_ampl=10.0_mean=0.0_noise=1.0.csv \
    -o artificial_sequences -c 8

# Generate sensortag data
pushd sensortag
python convert_acc_data.py
popd

# Convert sensortag data
python convert_to_sequences.py -i sensortag/converted/sensortag_x.csv \
    -o sensortag_sequences -c 100


