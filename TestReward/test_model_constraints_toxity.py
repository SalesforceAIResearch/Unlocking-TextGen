import json
import math
import argparse
import torch
import logging
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, LlamaTokenizer,  AutoModelForCausalLM, AutoModelForSequenceClassification
from pathlib import Path
import csv

from accelerate import init_empty_weights, infer_auto_device_map
import tensor_parallel as tp
from accelerate import dispatch_model, infer_auto_device_map
from accelerate.utils import get_balanced_memory

def get_device_map(model_name, device, do_int8):
    if device == "a100-40g":
        return "auto"

    with init_empty_weights():
        config = AutoConfig.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_config(config)

    d = {0: "18GiB"}
    for i in range(1, 6):
        d[i] = "26GiB"
    device_map = infer_auto_device_map(
        model, max_memory=d, dtype=torch.int8 if do_int8 else torch.float16,
        no_split_module_classes=["BloomBlock", "OPTDecoderLayer", "LLaMADecoderLayer", "LlamaDecoderLayer"]
    )
    print(device_map)
    del model
    return device_map



from typing import Iterable, Optional, Tuple

from torch import Tensor
from torch.nn import functional as F

import math

logger = logging.getLogger(__name__)

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

    parser.add_argument("--device", type=str, choices=["a100-40g", "v100-32g"], default="a100-40g")
    parser.add_argument("--do_int8", action="store_true")

    parser.add_argument('--mc', action="store_true", help="multiple constaints are separated.")

    parser.add_argument('--alpha', type=float, default=0.05,
                        help="decay factor for score in looking ahead")

    args = parser.parse_args()

    print(f"Decoding with: {args.model_name}")
 
    if 'alpaca' in args.model_name:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, unk_token="<unk>", bos_token="<s>", eos_token="</s>")
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    #tokenizer = AutoTokenizer.from_pretrained('gpt2-large')
    
    period_id = [tokenizer.convert_tokens_to_ids('.')]
    period_id.append(tokenizer.convert_tokens_to_ids('Ġ.'))
    eos_ids = [tokenizer.eos_token_id] + period_id
    PAD = tokenizer.eos_token # tokenizer.convert_tokens_to_ids('<pad>')
    PAD_ID = tokenizer.eos_token_id

    input_lines_A = []
    input_lines_B = []
    #goal_lines = []
  
    if 'nli' in args.model_name:
        goal_line = 'It will be a polite, respectful, and reasonable comment.'
        #goal_line = 'This will be a rude, disrespectful, or unreasonable comment.'
    else:
        # a rude, disrespectful, or unreasonable comment
        #goal_line = tokenizer.tokenize( tokenizer.eos_token + 'This will be a polite, respectful, and reasonable comment.')
        goal_line = tokenizer.tokenize( tokenizer.eos_token + 'This will be a rude, disrespectful, or unreasonable comment.')
        if args.mc: 
           goal_line = []
           goal_line.append( tokenizer.tokenize(tokenizer.eos_token + 'It will be a polite, respectful, and reasonable comment.'))
           goal_line.append(tokenizer.tokenize( tokenizer.eos_token + 'This will be a rude, disrespectful, or unreasonable comment.'))
        
           max_goal_len = max(len(x) for x in goal_line)
           goal_line = [x + [tokenizer.eos_token] * (max_goal_len - len(x)) for x in goal_line]
           goal_line = [ tokenizer.convert_tokens_to_ids(x)  for x in goal_line]


    f =  open('../data/toxicity/human_eval_toxicity.csv', mode= 'r')
    csvFile = csv.DictReader(f)
    for line in csvFile:
        if (len(line['senta']) > 1) and (len(line['sentb']) >1) and (len(line['toxa']) >1) and (len(line['toxb']) >1):
           #print(line['toxa'], line['toxb'])
           a = line['senta'].split()[0]
           b = line['sentb'].split()[0]
           if a==b:
               continue

           if 'nli' not in args.model_name:
              if float(line['toxa']) < float(line['toxb']):
                  input_lines_A.append(tokenizer.tokenize( line['sentprefix'] + b))
                  input_lines_B.append(tokenizer.tokenize( line['sentprefix'] + a))
              elif float(line['toxa']) > float(line['toxb']):
                  input_lines_B.append(tokenizer.tokenize( line['sentprefix'] + b))
                  input_lines_A.append(tokenizer.tokenize( line['sentprefix'] + a))

           else:
              if float(line['toxa']) < float(line['toxb']):
                   input_lines_A.append(line['sentprefix'] +' ' + a)
                   input_lines_B.append(line['sentprefix'] + ' ' +b)
              elif float(line['toxa']) > float(line['toxb']):
                  input_lines_B.append(line['sentprefix'] + ' ' +a)
                  input_lines_A.append(line['sentprefix'] + ' ' +b)



    total_batch = math.ceil(len(input_lines_A) / args.batch_size)
    next_i = 0

    print(len(input_lines_A), len(input_lines_B))
    print(input_lines_A[:1])
    print(input_lines_B[:1])
    #print(goal_lines[:10])

    if 'nli' in args.model_name:
        model=AutoModelForSequenceClassification.from_pretrained(args.model_name)

    else:
        """
        device_map = get_device_map(args.model_name, args.device, args.do_int8)
        model = AutoModelForCausalLM.from_pretrained(
           args.model_name,
           device_map=device_map,
           torch_dtype=torch.int8 if args.do_int8 else torch.float16,
           low_cpu_mem_usage=device_map is not None,
           load_in_8bit=args.do_int8,
           trust_remote_code=True,
        )
        """
        model = AutoModelForCausalLM.from_pretrained(
           args.model_name,
           device_map='auto',
           torch_dtype=torch.bfloat16,
           offload_folder="offload", offload_state_dict = True,
           load_in_8bit=args.do_int8,
           trust_remote_code=True,
        )
        #model = tp.tensor_parallel(model, [i for i in range(torch.cuda.device_count())])
        
        """
        model=AutoModelForCausalLM.from_pretrained(args.model_name, trust_remote_code=True)
        """
    
    torch.cuda.empty_cache()
    model.eval()
    #if ('nli' not in args.model_name):
    #    model = model.to('cuda').half()

    acc = []
    with tqdm(total=total_batch) as pbar:
        while next_i < len(input_lines_A):
            
            
            if next_i + args.batch_size > len(input_lines_A):
                step_len = len(input_lines_A)- next_i
            else:
                step_len =  args.batch_size

            goal_chunk =[ goal_line for _ in range(step_len)] #goal_lines[next_i:next_i + args.batch_size]
            #step_len= len(goal_chunk)

                   
            if 'nli' in args.model_name:
                buf_a = input_lines_A[next_i:next_i + step_len]
                buf_b = input_lines_B[next_i:next_i + step_len]
                next_i += step_len
                for premise1, premise2, hypothesis in zip(buf_a, buf_b, goal_chunk):
                   #print(premise1, premise2, hypothesis)
                   if (args.model_name=='microsoft/deberta-xlarge-mnli') or (args.model_name=='cross-encoder/nli-roberta-base'):
                       tokenized_input_seq_pair = tokenizer.encode_plus(premise1, hypothesis, max_length=256,return_token_type_ids=True, truncation=True)
                       input_ids = torch.Tensor(tokenized_input_seq_pair['input_ids']).long().unsqueeze(0)
                       output = model(input_ids)
                       #print(output[0])
                       # 0: contra; 1: entail
                       
                       #l2 = output[0].softmax(1)[0,1]
                       l1 = 1 - output[0].softmax(1)[0,0]
                       
                       #entail -contra
                       #l1 = 1-2*output[0].softmax(1)[0,0]
                       
                       tokenized_input_seq_pair = tokenizer.encode_plus(premise2, hypothesis, max_length=256,return_token_type_ids=True, truncation=True)
                       input_ids = torch.Tensor(tokenized_input_seq_pair['input_ids']).long().unsqueeze(0)
                       output = model(input_ids)
                       #l2 = output[0].softmax(1)[0,1]
                       # non-entail

                       l2 = 1 - output[0].softmax(1)[0,0]
                       #non-contral -contra
                       #l2 = 1 - 2*output[0].softmax(1)[0,0]

                   elif args.model_name=='facebook/bart-large-mnli':
                       x = tokenizer.encode(premise1, hypothesis, return_tensors='pt', truncation_strategy='only_first')
                       logits = model(x)[0]
                       entail_contradiction_logits = logits[:,[0,2]]
                       probs = entail_contradiction_logits.softmax(dim=1)
                       l1 = probs[:,1]

                       x = tokenizer.encode(premise2, hypothesis, return_tensors='pt', truncation_strategy='only_first')
                       logits = model(x)[0]
                       entail_contradiction_logits = logits[:,[0,2]]
                       probs = entail_contradiction_logits.softmax(dim=1)
                       l2 = probs[:,1]

                   if l1 >= l2:
                       acc.append(1)
                   else:
                       acc.append(0)

                   #print(acc) 

            else:

                   buf_a = input_lines_A[next_i:next_i + step_len]
                   buf_a = [ tokenizer.convert_tokens_to_ids(x) for x in buf_a]
                   max_goal_len = max([len(x) for x in buf_a])
                   buf_a = [x + [PAD_ID] * (max_goal_len - len(x)) for x in buf_a]
                   input_ids_a = torch.stack([torch.from_numpy(np.array(x)) for x in buf_a]) 
                   input_ids_a = input_ids_a.to('cuda')

                   buf_b = input_lines_B[next_i:next_i + step_len]
                   buf_b = [ tokenizer.convert_tokens_to_ids(x) for x in buf_b]
                   max_goal_len = max([len(x) for x in buf_b])
                   buf_b = [x + [PAD_ID] * (max_goal_len - len(x)) for x in buf_b]
                   input_ids_b = torch.stack([torch.from_numpy(np.array(x)) for x in buf_b])             
                   input_ids_b = input_ids_b.to('cuda')
           
                   if args.mc:
                       max_goal_len = max([len(x0) for x in goal_chunk for x0 in x])
                       goal_buf = [x0 + [PAD_ID] * (max_goal_len - len(x0)) for x in goal_chunk for x0 in x]
                       #look_goals = torch.stack([torch.from_numpy(np.array(x)) for x in goal_buf])
                       #look_goals= look_goals.view(look_goals.size(0), -1, max_goal_len)

                   else:
                       goal_chunk = [ tokenizer.convert_tokens_to_ids(x) for x in goal_chunk]
                       max_goal_len = max([len(x) for x in goal_chunk])
                       goal_buf = [ x + [PAD_ID] * (max_goal_len - len(x)) for x in goal_chunk]
           
   
                   look_goals = torch.stack([torch.from_numpy(np.array(x)) for x in goal_buf])
                   if args.mc:
                       look_goals= look_goals.view(step_len, -1, max_goal_len)
                   look_goals = look_goals.to('cuda')
      

                   next_i += step_len
                   
                   score_a = evalate_reverse_self(self=model, input_ids=input_ids_a,look_goals=look_goals, batch_size=step_len)
                   score_b = evalate_reverse_self(self=model, input_ids=input_ids_b,look_goals=look_goals, batch_size=step_len)
            
                   # for yes or no
                   #score_a = evalate_reverse_self2(self=model, input_ids=input_ids_a, look_goals=look_goals, pad_token_id=PAD_ID)
                   #score_b = evalate_reverse_self2(self=model, input_ids=input_ids_b, look_goals=look_goals, pad_token_id=PAD_ID)

                   #result = torch.gt(score_a, score_b)
                   result = torch.ge(score_a, score_b)
                   #print(score_a.size(), score_b.size(), result.size())
                   acc = acc + result.cpu().numpy().tolist()
           
                   pbar.update(1)

    acc = sum(acc)/len(acc)
    print(acc)



def evalate_reverse_self2(
        self,
        input_ids,
        look_goals,
        pad_token_id,
):
    """
    look_goals are the same for all input for this function
    """
    #look_goals = look_goals[:, 1,:]
    if look_goals.dim()==2:
        look_goals = look_goals.unsqueeze(1)

    batch_size = input_ids.size(0)
    total_constraint = look_goals.size(1)
    look_ahead_scores = 0

    flag =1
    for ii in range(total_constraint):
       #phrase_input_ids = input_ids.new(look_goals[ii])[None, :].expand(batch_size, -1)
       phrase_input_ids = torch.cat([input_ids, look_goals[:, ii]], dim=-1)

       cur_len = input_ids.size(1)
       with torch.no_grad():
           follow_logits = self(input_ids=phrase_input_ids, labels=phrase_input_ids)[1]

       ## shift logit token < n predict n
       follow_logits = follow_logits[:, cur_len:,]
       follow_log_prob = F.log_softmax(follow_logits, dim=-1)

       ## skip SEP symbole
       target_l = look_goals[:, ii, 1:].shape[-1]
       #look_ahead_score = follow_log_prob[:,torch.arange(target_l), look_goals[:, ii, 1:]]
        
       look_ahead_score = torch.gather(follow_log_prob, 2, look_goals[:, ii, 1:].unsqueeze(2)).squeeze(2)
       #print(look_ahead_score.size())
       mask = look_goals[:, ii, 1:].ne(pad_token_id).float()
       #print(mask.size())
       look_ahead_scores = look_ahead_scores + flag* torch.sum(look_ahead_score*mask, dim=-1) /torch.sum(mask, dim=-1)    
       flag = flag * (-1)
    #print(look_ahead_scores.size())
    return look_ahead_scores


def evalate_reverse_self(
        self,
        input_ids,
        look_goals,
        batch_size,
):

    #phrase_input_ids = input_ids.new(look_goals)[None, :].expand(batch_size, -1)
    phrase_input_ids = torch.cat([input_ids, look_goals], dim=-1)
    cur_len = input_ids.size(1)
    with torch.no_grad():
         follow_logits = self(input_ids=phrase_input_ids, labels=phrase_input_ids)[1]

    ## shift logit token < n predict n, skip eos
    #follow_logits = follow_logits[:, cur_len-1:,]
    
    # new version
    follow_logits = follow_logits[:, cur_len:,]
    follow_log_prob = F.log_softmax(follow_logits, dim=-1)

    target_l = look_goals.shape[-1]
    #look_ahead_scores = follow_log_prob[:, torch.arange(target_l), look_goals]
    look_ahead_scores = torch.gather(follow_log_prob, 2, look_goals[:, 1:].unsqueeze(2)).squeeze(2)
    
    # old
    #look_ahead_scores = torch.gather(follow_log_prob, 2, look_goals.unsqueeze(2)).squeeze(2)
    #print(look_ahead_scores.size(), look_ahead_scores[0])

    ## add first transitioin score
    #look_ahead_scores= torch.cat((log_prob[:,-1].unsqueeze(1), look_ahead_scores), dim=-1)
    #for i in range(batch_size):
    #       look_ahead_scores.append(follow_log_prob[i]
    look_ahead_scores = torch.mean(look_ahead_scores, dim=-1)
    #print(look_ahead_scores)
    return look_ahead_scores

if __name__ == "__main__":
    main()
