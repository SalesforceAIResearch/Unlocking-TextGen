#!/usr/bin/env bash

export PYTHONPATH='**'

DATA_DIR='../dataset/commongen'
SPLIT='test'



DEVICES=$1
MODEL_RECOVER_PATH=$2
alpha=$3

OUTPUT_FILE=A_star_new_Decoding_alpha${alpha}_${SPLIT}_prune5000_beam5

# neurologic with greedy look-ahead
CUDA_VISIBLE_DEVICES=${DEVICES} python -u NEUROLOGIC.py --model_name ${MODEL_RECOVER_PATH} \
  --input_path ${DATA_DIR}/${SPLIT}.txt --output_file ${OUTPUT_FILE} \
  --constraint_file ${DATA_DIR}/constraint/${SPLIT}.constraint.json \
  --key_constraint_file ${DATA_DIR}/constraint/${SPLIT}_key.constraint.json \
  --batch_size 8 --beam_size 5 --max_tgt_length 48 --min_tgt_length 5 \
  --ngram_size 3 --length_penalty 0.2 \
  --prune_factor 5000 --sat_tolerance 2 \
  --look_ahead_step 5  --alpha ${alpha} --look_ahead_width 1


