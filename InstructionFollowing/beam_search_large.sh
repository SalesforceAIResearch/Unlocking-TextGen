#!/usr/bin/env bash

DATA_DIR='../dataset'
DEVICES=$1
SPLIT='test' # 'test'
MODEL_RECOVER_PATH=$3  #'tiiuae/falcon-7b-instruct' # 'tiiuae/falcon-7b-instruct'
OUTPUT_FILE=CommonGen_${SPLIT}_$2
alpha=$2
# run decoding, max_length :64 before, length peanalty 0.2, --lookahead, beam_size 20
#CUDA_VISIBLE_DEVICES=${DEVICES} 
python beam_search_large.py --lookahead    --alpha ${alpha}   --model_name ${MODEL_RECOVER_PATH} \
  --input_path ${DATA_DIR}/commongen/${SPLIT}.txt --output_file ${OUTPUT_FILE} \
  --batch_size 8 --beam_size 20 --max_tgt_length 48 --min_tgt_length 5 \
  --ngram_size 3 --length_penalty 0.2
