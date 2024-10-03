#!/usr/bin/env bash

DATA_DIR='../dataset'
DEVICES=$1
SPLIT='dev'  # 'dev', dev.prefix, dev.prefix2, # 'test'
MODEL_RECOVER_PATH=$2  #  'microsoft/deberta-xlarge-mnli', 'tiiuae/falcon-7b-instruct' # 'tiiuae/falcon-7b-instruct'
OUTPUT_FILE=KeyWords_Test_${SPLIT}
# run decoding
CUDA_VISIBLE_DEVICES=${DEVICES} python test_model_constraints.py  --model_name ${MODEL_RECOVER_PATH} \
  --input_path ${DATA_DIR}/commongen/${SPLIT}.txt --output_file ${OUTPUT_FILE} \
  --batch_size 1 --beam_size 20 --max_tgt_length 48 --min_tgt_length 5 \
  --ngram_size 3 --length_penalty 0.2
