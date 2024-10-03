# LLM-Constrained-Decoding (LLM Constrained Decoding with A\*)
Official code and data release for [Unlocking Anticipatory Text Generation: A Constrained Approach for
Large Language Models Decoding](https://arxiv.org/pdf/2312.06149), accepted by EMNLP 2024.


## Install

The following command is used to install dependencies:
```
bash install.sh
```


## Future-Constrained Evaluation
* **Lexical-Constraint Satisfaction Evaluation**
```
test_model_constraints.sh
```

* **Toxicity-Constraint Satisfaction Evaluation**
```
test_model_constraints_toxity.sh
```

* **Factual-Correctness-Satisfaction Evaluation**
```
test_model_constraints_true.sh
test_model_constraints_true_YoN.sh

```


## Instruction Following
Go to the folder **InstructionFollowing**

* **NEUROLOGIC (adapted version for LLM)**
```
NEUROLOGIC.sh
```

* **Greedy, Beam-Search, Our**
```
beam_search_large.sh
```
In the above script, different settings can be used.

Greddy
```
beam_size=1
```

Beam-Search
```
beam_size=20
```

Our:
```
beam_size=20
choose alpha from [0.1, 10]
```

* **Evaluation**

Evaluation script is from [CommonGen](https://github.com/INK-USC/CommonGen/tree/master/evaluation) for automatic metrics.

## Toxicity Reduction 
Go to the folder **Toxicity**

```
toxity_script.sh
```

* **Evaluation**

Perspective API is used for toxicity evaluation.


## Factual QA
Go to the folder **FactaulQA**


* **main file**
```
beam_search_eli5_large.py
```

* **Claims as Constraints**

```
beam_search_eli5_large_restricted_claims.sh
```
* **Evaluation**

Evaluation script is from [ALCE](https://github.com/princeton-nlp/ALCE). We did not truncate the answer by the first newline.

## Citation
```
@inproceedings{
2024unlocking,
title={Unlocking Anticipatory Text Generation: A Constrained Approach for Large Language Models Decoding},
author={Lifu Tu and Semih Yavuz and Jin Qu and Jiacheng Xu and Rui Meng and Caiming Xiong and Yingbo Zhou},
booktitle={The 2024 Conference on Empirical Methods in Natural Language Processing},
year={2024},
url={https://openreview.net/forum?id=oeMJredL7n}
}
```
