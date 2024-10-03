#!/usr/bin/env bash

DATA_DIR='../dataset'
DEVICES=$1
SPLIT='test' # 'test'
MODEL_RECOVER_PATH=$2 # 'tiiuae/falcon-7b-instruct'
alpha=$3
beam=$4
task=$5

OUTPUT_FILE=${SPLIT}_length_penalty1_beam_${beam}_clip0.3
# run decoding, max_length :64 before, length peanalty 0.2, --lookahead, --YoN,  --claim  ,--nonICL, --quick_test 50, --use_shorter 'summary'
CUDA_VISIBLE_DEVICES=${DEVICES} python -u beam_search_eli5_large.py  --task_name ${task}    --split ${SPLIT} --OnlyClaim   --lookahead  --nonICL --alpha ${alpha}   --model_name ${MODEL_RECOVER_PATH} \
  --output_file ${OUTPUT_FILE}  --beam_size $beam --max_tgt_length 256 --min_tgt_length 5 \
  --ngram_size 3
