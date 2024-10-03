import json
import math
import argparse
import torch
import logging
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path

import tensor_parallel as tp


logger = logging.getLogger(__name__)

from base_generation import generate

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", type=str, help="pretrained language model to use")
    parser.add_argument("--input_path", type=str, help="path of input file")
    parser.add_argument("--output_file", type=str, help="output file")

    parser.add_argument('--batch_size', type=int, default=256,
                        help="Batch size for decoding.")
    parser.add_argument('--beam_size', type=int, default=10,
                        help="Beam size for searching")
    parser.add_argument('--max_tgt_length', type=int, default=100,
                        help="maximum length of decoded sentences")
    parser.add_argument('--min_tgt_length', type=int, default=0,
                        help="minimum length of decoded sentences")
    parser.add_argument('--ngram_size', type=int, default=3,
                        help='all ngrams can only occur once')
    parser.add_argument('--length_penalty', type=float, default=0.6,
                        help="length penalty for beam search")
    parser.add_argument('--lookahead', action="store_true")

    parser.add_argument('--mc', action="store_true", help="multiple constaints are separated.")

    parser.add_argument('--Addkeys', action="store_true", help="whether extra keys concetps are include candiatate.")

    parser.add_argument('--do_sample', action="store_true", help="Whether using sampling")

    parser.add_argument('--alpha', type=float, default=0.05,
                        help="decay factor for score in looking ahead")

    args = parser.parse_args()

    args.output_file = args.model_name.split('/')[-1] +'_answer_' +  args.output_file
    print(f"Decoding with: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    #model = AutoModelForCausalLM.from_pretrained(args.model_name, trust_remote_code=True)

    model = AutoModelForCausalLM.from_pretrained(
           args.model_name,
           device_map='auto',
           torch_dtype=torch.bfloat16,
           offload_folder="offload", offload_state_dict = True,
           trust_remote_code=True,
        )
    if torch.cuda.device_count() > 1:
        model = tp.tensor_parallel(model, [i for i in range(torch.cuda.device_count())])
    

    torch.cuda.empty_cache()
    model.eval()
    #model = model.to('cuda').half()


    period_id = [tokenizer.convert_tokens_to_ids('.')]
    period_id.append(tokenizer.convert_tokens_to_ids('Ġ.'))
    eos_ids = [tokenizer.eos_token_id] + period_id
    PAD = tokenizer.eos_token # tokenizer.convert_tokens_to_ids('<pad>')
    PAD_ID = tokenizer.eos_token_id

    input_lines = []
    goal_lines = []
    Key_lines = []
    with open(args.input_path) as fin:
        for line in fin.read().splitlines():
            ## concepts, or words for word order
            if "Llama-2" in args.model_name:
                input_lines.append("Write a sentence with these words:" + line.split('=')[0].strip() + '.\nAnswer:')
            else:
                input_lines.append("Write a sentence with these words:" + line.split('=')[0].strip() + '.')
            if args.mc:
                C = []
                allconstraint = line.split('=')[0].strip().split()
                l = len(allconstraint)//2

                # concepts, or words for word order
                C.append(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(PAD + ' This will be a sentence with these words:'+ ' '.join(allconstraint[:l]) + '.')))
                C.append(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(PAD + ' This will be a sentence with these words:'+ ' '.join(allconstraint[l:]) + '.')))
                goal_lines.append(C)
            else:
                goal_lines.append(tokenizer.convert_tokens_to_ids(tokenizer.tokenize( PAD + ' This will be a sentence with these concepts:' + line.split('=')[0].strip()+ '.' )) )
            #choice_tokens = [tokenizer.encode(choice, add_special_tokens=False)[0] for choice in line.split('=')[0].strip().split()]
            choice_tokens = tokenizer.encode(line.split('=')[0].strip(), add_special_tokens=False)

            Key_lines.append(choice_tokens)
    
    #o_input_lines = [x for x in input_lines]
    input_lines = [tokenizer.tokenize(x) for x in input_lines]

    input_lines = sorted(list(enumerate(input_lines)),
                         key=lambda x: len(x[1]))
    output_lines = [""] * len(input_lines)
  
   
    total_batch = math.ceil(len(input_lines) / args.batch_size)
    next_i = 0

 
    with tqdm(total=total_batch) as pbar:
        while next_i < len(input_lines):

            full_chunk = input_lines[next_i:next_i + args.batch_size]
            prompt_tokens_num = sorted(set([len(x[1]) for x in full_chunk]))
            step_len = args.batch_size
            if len(prompt_tokens_num) > 1:
                step_len = len([x for x in full_chunk if len(x[1]) == prompt_tokens_num[0]])

            #print(prompt_tokens_num[0], step_len)
            _chunk = input_lines[next_i:next_i + step_len]
            buf_id = [x[0] for x in _chunk]
            #print(buf_id)
            buf = [x[1] for x in _chunk]
        

            #goal_chunk = goal_lines[next_i:next_i + step_len]
            goal_chunk = [goal_lines[ii] for ii in buf_id]
            
            if args.mc:
                max_goal_len = max([len(x0) for x in goal_chunk for x0 in x])
                goal_buf = [x0 + [PAD_ID] * (max_goal_len - len(x0)) for x in goal_chunk for x0 in x]
                
            else:
                max_goal_len = max([len(x) for x in goal_chunk])
                goal_buf = [x + [PAD_ID] * (max_goal_len - len(x)) for x in goal_chunk]


            key_lines = [Key_lines[ii] for ii in buf_id]
            max_key_len = max([len(x) for x in key_lines])
            key_lines = [x + [PAD_ID] * (max_key_len - len(x)) for x in key_lines]

            next_i += step_len

            input_ids = torch.stack([torch.from_numpy(np.array(tokenizer.convert_tokens_to_ids(x))) for x in buf])
            #print(goal_buf)
            input_ids = input_ids.to('cuda')
            

            if args.lookahead:
                look_goals = torch.stack([torch.from_numpy(np.array(x)) for x in goal_buf])
                #print(look_goals.size())
                if args.mc:
                    # (batch size, num constraints, length)
                    look_goals= look_goals.view(look_goals.size(0), -1, max_goal_len)
                look_goals = look_goals.to('cuda')
            else:
                look_goals = None
            
            if args.alpha==0:
                look_goals = None

            if args.Addkeys:
                key_lines = torch.stack([torch.from_numpy(np.array(x)) for x in key_lines])
                key_lines = key_lines.to('cuda')
            else:
                key_lines = None

            #print('key lines', key_lines[0])
            ## --lifu
            outputs, _, _ = generate(self=model, input_ids=input_ids,
                                     #attention_mask=attention_mask,
                                     look_goals=look_goals,
                                     do_sample=args.do_sample,
                                     key_words = key_lines,
                                     alpha = args.alpha,
                                     pad_token_id=tokenizer.eos_token_id,
                                     #eos_token_id=tokenizer.eos_token_id,
                                     min_length=args.min_tgt_length+ prompt_tokens_num[0],
                                     max_length=args.max_tgt_length+ prompt_tokens_num[0],
                                     num_beams=args.beam_size,
                                     no_repeat_ngram_size=args.ngram_size,
                                     length_penalty=args.length_penalty)
            
            prompt = [tokenizer.convert_tokens_to_string(x) for x in buf]

            # use tokenizer.eos_token_id instead of '<|endoftext|>'
            output_sequences = [tokenizer.decode(o).split(tokenizer.eos_token)[0].split(prompt[i])[-1].strip()
                                for i, o in enumerate(outputs)]
            #print(output_sequences)
            for i in range(len(buf)):
                output_lines[buf_id[i]] = output_sequences[i].replace("\n", " ")
            #print(output_sequences) 
            pbar.update(1)
    
    with open(args.output_file, "w", encoding="utf-8") as fout:
        for l in output_lines:
            fout.write(l)
            fout.write("\n")
  


if __name__ == "__main__":
    main()
