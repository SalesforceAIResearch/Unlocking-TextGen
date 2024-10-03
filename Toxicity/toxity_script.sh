#python sample_toxity_llm.py  tiiuae/falcon-7b-instruct 1k -100
#python sample_toxity_llm.py  tiiuae/falcon-7b-instruct 200 -50
#python sample_toxity_llm.py  tiiuae/falcon-7b-instruct 1k -1000

#CUDA_VISIBLE_DEVICES=0,1 python sample_toxity_llm.py  /export/share/lifu/llama_hf/Llama-2-13b-chat-hf   10k 0 
CUDA_VISIBLE_DEVICES=0,1 python sample_toxity_llm.py  /export/share/lifu/llama_hf/Llama-2-13b-chat-hf   1k 1 
CUDA_VISIBLE_DEVICES=0,1 python sample_toxity_llm.py  /export/share/lifu/llama_hf/Llama-2-13b-chat-hf   1k 10 
CUDA_VISIBLE_DEVICES=0,1 python sample_toxity_llm.py  /export/share/lifu/llama_hf/Llama-2-13b-chat-hf   1k 100 



#python sample_toxity_llm.py chavinlo/alpaca-native   1k -100
#python sample_toxity_llm.py chavinlo/alpaca-native   1k -1000

#python sample_toxity_llm.py chavinlo/alpaca-native   200 5

#python sample_toxity_llm.py chavinlo/alpaca-native   200 0
#python sample_toxity_llm.py chavinlo/alpaca-native   200 1
#python sample_toxity_llm.py chavinlo/alpaca-native   200 5
#python sample_toxity_llm.py chavinlo/alpaca-native   200 -1
#python sample_toxity_llm.py chavinlo/alpaca-native   200 -10
#python sample_toxity_llm.py chavinlo/alpaca-native   200 -100
#python sample_toxity_llm.py chavinlo/alpaca-native   200 -200
#python sample_toxity_llm.py chavinlo/alpaca-native   200 -0.5
