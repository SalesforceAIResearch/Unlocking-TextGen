import json
import math
import argparse
import torch
import logging
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, LlamaForCausalLM
from pathlib import Path
from datasets import load_dataset
import re
import json
import time
import os
#import tensor_parallel as tp

from accelerate import dispatch_model, infer_auto_device_map
from accelerate.utils import get_balanced_memory


import random
random.seed(1)
np.random.seed(1)

logger = logging.getLogger(__name__)


from generate import generate

def make_doc_prompt(doc, doc_id, doc_prompt, use_shorter=None):
    # For doc prompt:
    # - {ID}: doc id (starting from 1)
    # - {T}: title
    # - {P}: text
    # use_shorter: None, "summary", or "extraction"

    text = doc['text']
    # some doc do not have "summary", or "extraction"
    if (use_shorter is not None) and (use_shorter in doc):
        text = doc[use_shorter]
    return doc_prompt.replace("{T}", doc["title"]).replace("{P}", text).replace("{ID}", str(doc_id+1))


def get_shorter_text(item, docs, ndoc, key):
    doc_list = []
    for item_id, item in enumerate(docs):
        if key not in item:
            if len(doc_list) == 0:
                # If there aren't any document, at least provide one (using full text)
                item[key] = item['text']
                doc_list.append(item)
            logger.warn(f"No {key} found in document. It could be this data do not contain {key} or previous documents are not relevant. This is document {item_id}. This question will only have {len(doc_list)} documents.")
            break
        if "irrelevant" in item[key] or "Irrelevant" in item[key]:
            continue
        doc_list.append(item)
        if len(doc_list) >= ndoc:
            break
    return doc_list


def make_demo(item, prompt, ndoc=None, doc_prompt=None, instruction=None, use_shorter=None, test=False):
    # For demo prompt
    # - {INST}: the instruction
    # - {D}: the documents
    # - {Q}: the question
    # - {A}: the answers
    # ndoc: number of documents to put in context
    # use_shorter: None, "summary", or "extraction"

    prompt = prompt.replace("{INST}", instruction).replace("{Q}", item['question'])
    if "{D}" in prompt:
        if ndoc == 0:
            prompt = prompt.replace("{D}\n", "") # if there is no doc we also delete the empty line
        else:
            doc_list = get_shorter_text(item, item["docs"], ndoc, use_shorter) if use_shorter is not None else item["docs"][:ndoc]
            text = "".join([make_doc_prompt(doc, doc_id, doc_prompt, use_shorter=use_shorter) for doc_id, doc in enumerate(doc_list)])
            docs =text
            prompt = prompt.replace("{D}", text)

    if not test:
        answer = "\n" + "\n".join(item["answer"]) if isinstance(item["answer"], list) else item["answer"]
        prompt = prompt.replace("{A}", "").rstrip() + answer
    else:
        prompt = prompt.replace("{A}", "").rstrip() # remove any space or \n

    return prompt, docs


def get_max_memory():
    """Get the maximum memory available for the current GPU for loading models."""
    free_in_GB = int(torch.cuda.mem_get_info()[0]/1024**3)
    max_memory = f'{free_in_GB-6}GB'
    n_gpus = torch.cuda.device_count()
    max_memory = {i: max_memory for i in range(n_gpus)}
    return max_memory

def load_model(model_name_or_path, dtype=torch.float16, int8=False, reserve_memory=10):
    # Load a huggingface model and tokenizer
    # dtype: torch.float16 or torch.bfloat16
    # int8: whether to use int8 quantization
    # reserve_memory: how much memory to reserve for the model on each gpu (in GB)

    # Llama: set up the root dir
    #open_source_models = ["llama", "alpaca", "vicuna", "oasst"]
    #if any([m in model_name_or_path for m in open_source_models]):
    #    model_name_or_path = os.path.join(os.environ["LLAMA_ROOT"], model_name_or_path)

    # Load the FP16 model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers_flash.modeling.modeling_llama_flash import LlamaForCausalLM as LlamaForCausalLMFlash
    logger.info(f"Loading {model_name_or_path} in {dtype}...")
    if int8:
        logger.warn("Use LLM.int8")
    start_time = time.time()

    #max_memory = {0: "34GIB", 1: "20GIB"} 
    """
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        device_map='auto',
        torch_dtype=dtype,
        #max_memory=get_max_memory(),
        load_in_8bit=int8,
    )
    """
    if 'falcon' in model_name_or_path:
        model = AutoModelForCausalLM.from_pretrained(
           model_name_or_path,
           device_map='auto',
           torch_dtype=torch.bfloat16,
           offload_folder="offload", offload_state_dict = True,
           trust_remote_code=True,
        )
    else:
        """
        model = LlamaForCausalLMFlash.from_pretrained(
            model_name_or_path, resume_download=True,
            torch_dtype=dtype, device_map="auto", offload_folder="offload", offload_state_dict = True,
        )
        """
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            device_map='auto',
            torch_dtype=dtype,
            offload_folder="offload",
            offload_state_dict = True,
        )
    
    """ 
    max_memory = {0: "20GIB", 1: "35GIB"}
    for i in range(2, torch.cuda.device_count()):
        max_memory[i] = "35GIB"
     
    max_memory = get_balanced_memory(
       model,
       max_memory=max_memory,
       no_split_module_classes=["BloomBlock", "OPTDecoderLayer", "LLaMADecoderLayer", "LlamaDecoderLayer"],
       dtype=dtype,
       low_zero=False,  #False,
    )
    
    #max_memory = {0: "34GIB", 1: "20GIB"}
    print(max_memory)

    device_map = infer_auto_device_map(
       model,
       max_memory=max_memory,
       no_split_module_classes=["BloomBlock", "OPTDecoderLayer", "LLaMADecoderLayer", "LlamaDecoderLayer"],
       dtype=dtype
    )
    print(device_map)
    model = dispatch_model(model, device_map=device_map)
    """ 
    #if torch.cuda.device_count() > 1:
    #    model = tp.tensor_parallel(model, [i for i in range(torch.cuda.device_count())])

    logger.info("Finish loading in %.2f sec." % (time.time() - start_time))
    
    """
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=False)

    # Fix OPT bos token problem in HF
    if "opt" in model_name_or_path:
        tokenizer.bos_token = "<s>"
    tokenizer.padding_side = "left"
    """
    return model



def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", type=str, help="pretrained language model to use")
    parser.add_argument("--input_path", type=str, help="path of input file")
    parser.add_argument("--output_file", type=str, help="output file")
    parser.add_argument("--prompt_file", type=str, help="Path to the prompt file")
    parser.add_argument("--use_shorter", type=str, default=None, help="Whether to use summary data or extraction data for documents. Option: None, `summary`, `extraction`")

    parser.add_argument("--quick_test", type=int, default=None, help="Quickly test a few examples")
    parser.add_argument("--no_doc_in_demo", type=bool, default=False, help="Whether to remove the documents in the demos")
    parser.add_argument("--fewer_doc_in_demo", type=bool, default=False, help="Whether to use fewer documents in the demos")
    parser.add_argument("--ndoc_in_demo", type=int, default=None, help="When using --fewer_doc_in_demo, use this to designate how many docs in demo")

    parser.add_argument("--ndoc", type=int,default=3, help="Number of documents")
    parser.add_argument("--shot", type=int, default=2, help="Number of ICL demonstrations")

    parser.add_argument('--batch_size', type=int, default=1,
                        help="Batch size for decoding.")
    parser.add_argument('--split', type=str, help="Split partion for decoding.")

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
    parser.add_argument('--repetition_penalty', type=float, default=1.0,
                        help="repetition penalty for repetive tokens")

    parser.add_argument('--lookahead', action="store_true")

    parser.add_argument('--claim', action="store_true", help="claims constaints are used.")

    parser.add_argument("--YoN", action="store_true", help="binary reward is used.")

    parser.add_argument('--mc', action="store_true", help="multiple constaints are separated.")
    parser.add_argument('--nonICL', action="store_true", help="no incontext learning.")

    parser.add_argument('--do_sample', action="store_true", help="Whether using sampling")

    parser.add_argument("--task_name", type=str, default='eli5',  help="task")

    parser.add_argument("--GuideFrquency", type=int, default=1, help="Frequency for Instrct constrained decoding")

    parser.add_argument("--OnlyClaim", action="store_true", help="Whether only inference only retrieved doc support by claims")

    parser.add_argument("--GeneratedClaim", action="store_true", help="Whether only inference with generated claims based on questions and docs")

    parser.add_argument("--IncludeQ_rank", action="store_true", help="Whether with question for ranking")

    parser.add_argument('--alpha', type=float, default=0.05,
                        help="decay factor for score in looking ahead")

    args = parser.parse_args()

    print(f"Decoding with: {args.model_name}")
    if 'alpaca' in args.model_name:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, unk_token="<unk>", bos_token="<s>", eos_token="</s>")
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)

    # Fix OPT bos token problem in HF
    if "opt" in args.model_name:
        tokenizer.bos_token = "<s>"
    tokenizer.padding_side = "left"
    
    if tokenizer.pad_token_id==None:
        tokenizer.pad_token_id=tokenizer.eos_token_id

    #tokenizer = AutoTokenizer.from_pretrained('gpt2-large')
    PAD = tokenizer.eos_token
    PAD_ID = tokenizer.eos_token_id

    

    if 'nli' in args.model_name:
        model=AutoModelForSequenceClassification.from_pretrained(args.model_name)

    elif ('chatglm2' in args.model_name):
        model = AutoModel.from_pretrained(args.model_name, trust_remote_code=True).bfloat16().cuda()
        #model = AutoModel.from_pretrained(args.model_name, trust_remote_code=True).half().cuda()
    else:
        model = load_model(args.model_name, dtype=torch.bfloat16)
        #model = load_model(args.model_name, dtype=torch.float16)
        
        #device_map = get_device_map(args.model_name, args.device, args.do_int8)
        #model = AutoModelForCausalLM.from_pretrained(
        #   args.model_name,
        #   device_map='auto',
        #   torch_dtype=torch.bfloat16,
        #   low_cpu_mem_usage=True,
        #   trust_remote_code=True,
        #)

    #stop = list(set( ["\n", "Ċ", "ĊĊ", "<0x0A>"])) # In Llama \n is <0x0A>; In OPT \n is Ċ
    #stop_token_ids = list(set([tokenizer.encode(stop_token, add_special_tokens=False)[0] for stop_token in stop] + [model.config.eos_token_id]))
    
    if args.YoN:
       ## for binary reward
        choices = ["Yes", "No"]
        choice_tokens = [tokenizer.encode(choice, add_special_tokens=False)[0] for choice in choices]
    else:
        choice_tokens = None
    #if "llama" in args.model_name:
    #        stop_token_ids.remove(tokenizer.unk_token_id)

    #model = AutoModelForCausalLM.from_pretrained(args.model_name, trust_remote_code=True)

    torch.cuda.empty_cache()
    model.eval()

    #head_prompt = 'Instruction: Write a high-quality answer for the given question using only the provided search results.\n\n'
    if args.task_name=='asqa':
        eval_data = json.load(open('../data/asqa_eval_gtr_top100_reranked_oracle.json'))
    
    elif args.task_name=='qampari':
        eval_data = json.load(open('../data/qampari_eval_gtr_top100_reranked_oracle.json'))

    elif args.task_name=='eli5':
        if args.GeneratedClaim:
            eval_data = json.load(open('../data/eli5_gpt4_examples_Noneeval_bm25_top100_reranked_oracle_withGenerateClaims.json'))
        else:
            eval_data = json.load(open('../data/eli5_eval_bm25_top100_reranked_oracle.json'))

    if args.quick_test is not None:
        eval_ids = np.random.choice(len(eval_data), args.quick_test, replace=False)
        eval_data = [eval_data[int(idx)] for idx in eval_ids]
    
    if args.OnlyClaim:
        Tmp_data = [item for item in eval_data]
        eval_data = []
        for item in Tmp_data:
            doc_searched = item['docs'][:args.ndoc]
            #flag = np.array([0,0,0])
            ll_len = len(doc_searched[0]['answers_found'])
            flag = np.array([0 for _ in range(ll_len)])
            for doc in doc_searched:
                flag = flag + np.array(doc['answers_found'])
            #if (flag[0]>0) and (flag[1]>0) and (flag[2]>0):
            #    eval_data.append(item)

            remove = 0
            for i in range(ll_len):
                if flag[i]==0:
                    remove=1
                    break
            if remove==0:
                eval_data.append(item)

    if args.task_name=='asqa':
        prompt_data = json.load(open('prompts/my_asqa.json'))
        #prompt_data = json.load(open('prompts/asqa_light_inst.json'))
    if args.task_name=='qampari':
        prompt_data = json.load(open('prompts/my_qampari_light_inst.json'))

    elif args.task_name=='eli5':
        if args.use_shorter is not None:
           prompt_data = json.load(open('prompts/eli5_sum_or_ext.json'))
        else:
           prompt_data = json.load(open('prompts/my_eli5.json'))
           #prompt_data = json.load(open('prompts/eli5_light_inst.json'))

    # Generate the demonstration part
    head_prompt = ""
    train_ids = np.random.choice(len(prompt_data["demos"]), args.shot, replace=False)
    for train_id in train_ids:
        train_item = prompt_data["demos"][train_id]
        ndoc = args.ndoc
        if args.no_doc_in_demo:
            ndoc = 0
        elif args.fewer_doc_in_demo:
            assert args.ndoc_in_demo is not None
            ndoc = args.ndoc_in_demo
        head_prompt += make_demo(
            train_item, prompt=prompt_data["demo_prompt"], ndoc=ndoc, doc_prompt=prompt_data["doc_prompt"],
            instruction=prompt_data["instruction"], use_shorter=args.use_shorter
        )[0]
        head_prompt += prompt_data["demo_sep"]
    #print(head_prompt)
    ICL = len(tokenizer(head_prompt)['input_ids'])
    #print(ICL)
    input_lines= []
    goal_lines = []

    if args.YoN:
        prefix_reward_prompts = tokenizer("Claim:")['input_ids']
        prefix_reward_prompts = torch.from_numpy(np.array(prefix_reward_prompts)).to('cuda')
    else:
        prefix_reward_prompts = None

    for idx, eval_item in enumerate(tqdm(eval_data)):     
        prompt, docs =  make_demo(
            eval_item, prompt=prompt_data["demo_prompt"], ndoc=args.ndoc, doc_prompt=prompt_data["doc_prompt"],
            instruction=prompt_data["instruction"], use_shorter=args.use_shorter,
            test=True
        )
        #print(head_prompt) 
        #print(prompt)
        #print("*"*100)
        #print(docs)
        prompt = head_prompt + prompt
        input_lines.append(tokenizer(prompt)['input_ids'])
        
        #print(prompt)
        if args.GeneratedClaim:
            g = ' '.join(eval_item["qdoc2claims"])
        else:
            if args.claim:
                 g = ' '.join(eval_item["claims"])
            else:
                 g = docs.strip()

        if args.YoN:
            YoN_prompts = f"Document: {g}\n\nQuestion: Is the above claim supported by the above document? Answer with Yes or No.\n\nAnswer:"
            # keep prfix for "Claim:" 
            #keep new line symbole after generated text
            goal_line = tokenizer( "\n\n" + YoN_prompts, add_special_tokens=False)['input_ids']

        else:
            # goal_line = tokenizer( PAD + g)['input_ids']
            #goal_line = tokenizer( g, add_special_tokens=False)['input_ids']
            goal_line = tokenizer( g)['input_ids']

        #print(prompt)
        #print('*'*100)
        #print(g)
        #print('f'*100)
        goal_lines.append( [PAD_ID] + goal_line)
        
    
        
    next_i = 0
    File = 'Take_name'+ str(args.task_name) +  "GenerateClaim" +  str(args.GeneratedClaim)+ 'GuideFrquency_'+ str(args.GuideFrquency)+"_OnlyClaimdoc_" + str(args.OnlyClaim)+ 'test_nonICL_' +  str(args.nonICL) + '_QuickTest_' + str(args.quick_test) + '_YoN_' + str(args.YoN) +'_claim_' +str(args.claim) + '_smmartPropt_' + str(args.use_shorter)  + '_mc_' +  str(args.mc) +'_alpha_' + str(args.alpha) + '_lookahead_'  + str(args.lookahead)  + args.model_name.split('/')[-1] + args.output_file
    #fd = open(File, 'w')

    with tqdm(total=len(input_lines)) as pbar:
       for input_line, goal_line, item in tqdm(zip(input_lines, goal_lines, eval_data), total=len(input_lines)):
                   
            inputs = torch.from_numpy(np.array(input_line)).unsqueeze(0)
            prompt_tokens_num = inputs.size(-1)
            #input_ids = torch.stack([torch.from_numpy(np.array(tokenizer.convert_tokens_to_ids(x))) for x in buf])
            input_ids = inputs.to('cuda')
           
            if args.IncludeQ_rank:
                ICL_i = ICL
            else:
                ICL_i = prompt_tokens_num

            

            if args.lookahead and (args.alpha!=0):
                look_goals= torch.from_numpy(np.array(goal_line)).unsqueeze(0)
                look_goals = look_goals.to('cuda')
            else:
                look_goals = None


            ## --lifu
            with torch.no_grad():
                outputs, _, _ = generate(self=model, input_ids=input_ids,
                                     #attention_mask=attention_mask,
                                     look_goals=look_goals,
                                     do_sample=args.do_sample,
                                     prefix_reward_prompts=prefix_reward_prompts,
                                     choice_tokens=choice_tokens,
                                     alpha = args.alpha,
                                     ICL_length=ICL_i,
                                     pad_token_id=tokenizer.eos_token_id,
                                     eos_token_id=tokenizer.eos_token_id,
                                     min_length=args.min_tgt_length+ prompt_tokens_num,
                                     max_length=args.max_tgt_length+ prompt_tokens_num,
                                     num_beams=args.beam_size,
                                     no_repeat_ngram_size=args.ngram_size,
                                     repetition_penalty=args.repetition_penalty,
                                     length_penalty=args.length_penalty, GuideFrquency=args.GuideFrquency)
            
            prompt = tokenizer.decode(input_line)
            #print('prompt: ', prompt)


            output_sequences = tokenizer.decode(outputs[0]).split(tokenizer.eos_token)[0].split(prompt)[-1].strip()
            #print(tokenizer.decode(outputs[0]))
            item['output'] = output_sequences
            print('*'*100)
            print(output_sequences)
        
            pbar.update(1)

    eval_data = {
        "args": args.__dict__,
        "data": eval_data,
    }

    if not os.path.exists("ALCEResultFinal"):
        os.makedirs("ALCEResultFinal")
    json.dump(eval_data, open("ALCEResultFinal/" + File + ".json", "w"), indent=4)



if __name__ == "__main__":
    main()
