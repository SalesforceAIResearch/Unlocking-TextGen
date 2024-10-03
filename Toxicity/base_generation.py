import logging
from typing import Iterable, Optional, Tuple

import torch
from torch import Tensor
from torch.nn import functional as F

import math
import numpy as np

from transformers import AutoTokenizer, AutoModelWithLMHead
#tokenizer = AutoTokenizer.from_pretrained('../PretrainModels/checkpoint-1800/')
#tokenizer = AutoTokenizer.from_pretrained('tiiuae/falcon-7b-instruct')

#TestModel=AutoModelWithLMHead.from_pretrained('../PretrainModels/ReverseModel/')
#TestModel=AutoModelWithLMHead.from_pretrained('gpt2-xl')
#TestModel.eval()
#TestModel = TestModel.to('cuda').half()

logger = logging.getLogger(__name__)



def Falcon_reorder_cache(self, past, batch_size, beam_idx):
        
        def _convert_to_standard_cache(
            past_key_value, batch_size):
            """
            Standardizes the format of the cache so as to match most implementations, i.e. to tuple(tuple([batch_size,
            num_heads, ...]))
            """
            batch_size_times_num_heads, head_dim, seq_length = past_key_value[0][0].shape
            num_heads = batch_size_times_num_heads // batch_size
            # key: [batch_size * num_heads, head_dim, seq_length] -> [batch_size, num_heads, head_dim, seq_length]
            # value: [batch_size * num_heads, seq_length, head_dim] -> [batch_size, num_heads, seq_length, head_dim]
            return tuple(
                (
                   layer_past[0].view(batch_size, num_heads, head_dim, seq_length),
                   layer_past[1].view(batch_size, num_heads, seq_length, head_dim),
                )
                for layer_past in past_key_value
            )

        def _convert_to_rw_cache(past_key_value):
            batch_size, num_heads, head_dim, seq_length = past_key_value[0][0].shape
            batch_size_times_num_heads = batch_size * num_heads
            # key:  [batch_size, num_heads, head_dim, seq_length] -> [batch_size * num_heads, head_dim, seq_length]
            # value: [batch_size, num_heads, seq_length, head_dim] -> [batch_size * num_heads, seq_length, head_dim]
            return tuple(
               (
                layer_past[0].view(batch_size_times_num_heads, head_dim, seq_length),
                layer_past[1].view(batch_size_times_num_heads, seq_length, head_dim),
               )
               for layer_past in past_key_value
            )

        standardized_past = _convert_to_standard_cache(past, batch_size=batch_size)
        # Get a copy of `beam_idx` on all the devices where we need those indices.
        device_to_beam_idx = {
            past_state.device: beam_idx.to(past_state.device) for layer_past in past for past_state in layer_past
        }
        reordered_past = tuple(
            (
                layer_past[0].index_select(0, device_to_beam_idx[layer_past[0].device]),
                layer_past[1].index_select(0, device_to_beam_idx[layer_past[0].device]),
            )
            for layer_past in standardized_past
        )
        return _convert_to_rw_cache(reordered_past)



# part is copied from https://huggingface.co/tiiuae/falcon-40b-instruct/blob/main/modelling_RW.py
def Falcon40b_reorder_cache(
        self, past, batch_size, beam_idx):
        """
        This function is used to re-order the `past_key_values` cache if [`~PreTrainedModel.beam_search`] or
        [`~PreTrainedModel.beam_sample`] is called. This is required to match `past_key_values` with the correct
        beam_idx at every generation step.
        Output shares the same memory storage as `past`.
        """
        def _convert_to_standard_cache(
            past_key_value, batch_size):
            """
            Standardizes the format of the cache so as to match most implementations, i.e. to tuple(tuple([batch_size,
            num_heads, ...]))
            """
            batch_size_times_num_heads, head_dim, seq_length = past_key_value[0][0].shape
            num_heads = batch_size_times_num_heads // batch_size
            # key: [batch_size * num_heads, head_dim, seq_length] -> [batch_size, num_heads, head_dim, seq_length]
            # value: [batch_size * num_heads, seq_length, head_dim] -> [batch_size, num_heads, seq_length, head_dim]
            return tuple(
                (
                   layer_past[0].view(batch_size, num_heads, head_dim, seq_length),
                   layer_past[1].view(batch_size, num_heads, seq_length, head_dim),
                )
                for layer_past in past_key_value
            )

        def _convert_to_rw_cache(
            past_key_value):
            batch_size, num_heads, head_dim, seq_length = past_key_value[0][0].shape
            batch_size_times_num_heads = batch_size * num_heads
            # key:  [batch_size, num_heads, head_dim, seq_length] -> [batch_size * num_heads, head_dim, seq_length]
            # value: [batch_size, num_heads, seq_length, head_dim] -> [batch_size * num_heads, seq_length, head_dim]
            return tuple(
                (
                layer_past[0].view(batch_size_times_num_heads, head_dim, seq_length),
                layer_past[1].view(batch_size_times_num_heads, seq_length, head_dim),
                )
                for layer_past in past_key_value
            )


        standardized_past = _convert_to_standard_cache(past, batch_size=batch_size)

        # Get a copy of `beam_idx` on all the devices where we need those indices.
        device_to_beam_idx = {
            past_state.device: beam_idx.to(past_state.device) for layer_past in past for past_state in layer_past
        }
        reordered_past = tuple(
            (
                layer_past[0].index_select(0, device_to_beam_idx[layer_past[0].device]),
                layer_past[1].index_select(0, device_to_beam_idx[layer_past[0].device]),
            )
            for layer_past in standardized_past
        )
        return _convert_to_rw_cache(reordered_past)


# copy from https://huggingface.co/transformers/v3.0.2/_modules/transformers/generation_utils.html


def calc_banned_bad_words_ids(prev_input_ids: Iterable[int], bad_words_ids: Iterable[int]) -> Iterable[int]:
    banned_tokens = []

    def _tokens_match(prev_tokens, tokens):
        if len(tokens) == 0:
            # if bad word tokens is just one token always ban it
            return True
        if len(tokens) > len(prev_input_ids):
            # if bad word tokens are longer then prev input_ids they can't be equal
            return False

        if prev_tokens[-len(tokens) :] == tokens:
            # if tokens match
            return True
        else:
            return False

    for prev_input_ids_slice in prev_input_ids:
        banned_tokens_slice = []

        for banned_token_seq in bad_words_ids:
            assert len(banned_token_seq) > 0, "Banned words token sequences {} cannot have an empty list".format(
                bad_words_ids
            )

            if _tokens_match(prev_input_ids_slice.tolist(), banned_token_seq[:-1]) is False:
                # if tokens do not match continue
                continue

            banned_tokens_slice.append(banned_token_seq[-1])

        banned_tokens.append(banned_tokens_slice)

    return banned_tokens



def calc_banned_ngram_tokens(prev_input_ids: Tensor, num_hypos: int, no_repeat_ngram_size: int, cur_len: int) -> None:
    """Copied from fairseq for no_repeat_ngram in beam_search"""
    if cur_len + 1 < no_repeat_ngram_size:
        # return no banned tokens if we haven't generated no_repeat_ngram_size tokens yet
        return [[] for _ in range(num_hypos)]
    generated_ngrams = [{} for _ in range(num_hypos)]
    for idx in range(num_hypos):
        gen_tokens = prev_input_ids[idx].tolist()
        generated_ngram = generated_ngrams[idx]
        for ngram in zip(*[gen_tokens[i:] for i in range(no_repeat_ngram_size)]):
            prev_ngram_tuple = tuple(ngram[:-1])
            generated_ngram[prev_ngram_tuple] = generated_ngram.get(prev_ngram_tuple, []) + [ngram[-1]]

    def _get_generated_ngrams(hypo_idx):
        # Before decoding the next token, prevent decoding of ngrams that have already appeared
        start_idx = cur_len + 1 - no_repeat_ngram_size
        ngram_idx = tuple(prev_input_ids[hypo_idx, start_idx:cur_len].tolist())
        return generated_ngrams[hypo_idx].get(ngram_idx, [])

    banned_tokens = [_get_generated_ngrams(hypo_idx) for hypo_idx in range(num_hypos)]
    return banned_tokens



def postprocess_next_token_scores(
        scores,
        input_ids,
        no_repeat_ngram_size,
        bad_words_ids,
        cur_len,
        min_length,
        max_length,
        eos_token_id,
        repetition_penalty,
        batch_size,
        num_beams,
    ):
        # repetition penalty (from CTRL paper https://arxiv.org/abs/1909.05858)
        if repetition_penalty != 1.0:
            ### remove
            #self.enforce_repetition_penalty_(
            #    scores, batch_size, num_beams, input_ids, repetition_penalty,
            #)
            for i in range(batch_size * num_beams):
              for previous_token in set(input_ids[i].tolist()):
                # if score < 0 then repetition penalty has to multiplied to reduce the previous token probability
                if scores[i, previous_token] < 0:
                    scores[i, previous_token] *= repetition_penalty
                else:
                    scores[i, previous_token] /= repetition_penalty


        # set eos token prob to zero if min_length is not reached
        if eos_token_id is not None and cur_len < min_length:
            scores[:, eos_token_id] = -float("inf")

        if no_repeat_ngram_size > 0:
            # calculate a list of banned tokens to prevent repetitively generating the same ngrams
            num_batch_hypotheses = batch_size * num_beams
            # from fairseq: https://github.com/pytorch/fairseq/blob/a07cb6f40480928c9e0548b737aadd36ee66ac76/fairseq/sequence_generator.py#L345
            banned_batch_tokens = calc_banned_ngram_tokens(
                input_ids, num_batch_hypotheses, no_repeat_ngram_size, cur_len
            )
            for i, banned_tokens in enumerate(banned_batch_tokens):
                scores[i, banned_tokens] = -float("inf")

        if bad_words_ids is not None:
            # calculate a list of banned tokens according to bad words
            banned_tokens = calc_banned_bad_words_ids(input_ids, bad_words_ids)

            for i, banned_tokens in enumerate(banned_tokens):
                scores[i, banned_tokens] = -float("inf")

        return scores


@torch.no_grad()
def generate(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        max_length: Optional[int] = None,
        min_length: Optional[int] = None,
        do_sample: Optional[bool] = None,
        early_stopping: Optional[bool] = None,
        num_beams: Optional[int] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        repetition_penalty: Optional[float] = None,
        bad_words_ids: Optional[Iterable[int]] = None,
        bos_token_id: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        length_penalty: Optional[float] = None,
        no_repeat_ngram_size: Optional[int] = None,
        num_return_sequences: Optional[int] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        decoder_start_token_id: Optional[int] = None,
        use_cache: Optional[bool] = None,
        look_goals: Optional[torch.LongTensor] = None,
        key_words: Optional[torch.LongTensor] = None,
        alpha: Optional[float] = None,
        **model_specific_kwargs
) -> torch.LongTensor:
    r""" Generates sequences for models with a LM head. The method currently supports greedy decoding, beam-search decoding, sampling with temperature, sampling with top-k or nucleus sampling.

    Adapted in part from `Facebook's XLM beam search code`_.

    .. _`Facebook's XLM beam search code`:
       https://github.com/facebookresearch/XLM/blob/9e6f6814d17be4fe5b15f2e6c43eb2b2d76daeb4/src/model/transformer.py#L529


    Parameters:

        input_ids: (`optional`) `torch.LongTensor` of shape `(batch_size, sequence_length)`
            The sequence used as a prompt for the generation. If `None` the method initializes
            it as an empty `torch.LongTensor` of shape `(1,)`.

        max_length: (`optional`) int
            The max length of the sequence to be generated.  Between `min_length` and infinity. Default to 20.

        min_length: (`optional`) int
            The min length of the sequence to be generated.  Between 0 and infinity. Default to 0.

        do_sample: (`optional`) bool
            If set to `False` greedy decoding is used. Otherwise sampling is used. Defaults to `False` as defined in `configuration_utils.PretrainedConfig`.

        early_stopping: (`optional`) bool
            if set to `True` beam search is stopped when at least `num_beams` sentences finished per batch. Defaults to `False` as defined in `configuration_utils.PretrainedConfig`.

        num_beams: (`optional`) int
            Number of beams for beam search. Must be between 1 and infinity. 1 means no beam search. Default to 1.

        temperature: (`optional`) float
            The value used to module the next token probabilities. Must be strictly positive. Default to 1.0.

        top_k: (`optional`) int
            The number of highest probability vocabulary tokens to keep for top-k-filtering. Between 1 and infinity. Default to 50.

        top_p: (`optional`) float
            The cumulative probability of parameter highest probability vocabulary tokens to keep for nucleus sampling. Must be between 0 and 1. Default to 1.

        repetition_penalty: (`optional`) float
            The parameter for repetition penalty. Between 1.0 and infinity. 1.0 means no penalty. Default to 1.0.

        pad_token_id: (`optional`) int
            Padding token. Default to specicic model pad_token_id or None if it does not exist.

        bos_token_id: (`optional`) int
            BOS token. Defaults to `bos_token_id` as defined in the models config.

        eos_token_id: (`optional`) int
            EOS token. Defaults to `eos_token_id` as defined in the models config.

        length_penalty: (`optional`) float
            Exponential penalty to the length. Default to 1.

        no_repeat_ngram_size: (`optional`) int
            If set to int > 0, all ngrams of size `no_repeat_ngram_size` can only occur once.
        bad_words_ids: (`optional`) list of lists of int
            `bad_words_ids` contains tokens that are not allowed to be generated. In order to get the tokens of the words that should not appear in the generated text, use `tokenizer.encode(bad_word, add_prefix_space=True)`.

        num_return_sequences: (`optional`) int
            The number of independently computed returned sequences for each element in the batch. Default to 1.

        attention_mask (`optional`) obj: `torch.LongTensor` of same shape as `input_ids`
            Mask to avoid performing attention on padding token indices.
            Mask values selected in ``[0, 1]``:
            ``1`` for tokens that are NOT MASKED, ``0`` for MASKED tokens.
            Defaults to `None`.

            `What are attention masks? <../glossary.html#attention-mask>`__

        decoder_start_token_id=None: (`optional`) int
            If an encoder-decoder model starts decoding with a different token than BOS.
            Defaults to `None` and is changed to `BOS` later.

        use_cache: (`optional`) bool
            If `use_cache` is True, past key values are used to speed up decoding if applicable to model. Defaults to `True`.

        model_specific_kwargs: (`optional`) dict
            Additional model specific kwargs will be forwarded to the `forward` function of the model.

    Return:

        output: `torch.LongTensor` of shape `(batch_size * num_return_sequences, sequence_length)`
            sequence_length is either equal to max_length or shorter if all batches finished early due to the `eos_token_id`

    Examples::

        tokenizer = AutoTokenizer.from_pretrained('distilgpt2')   # Initialize tokenizer
        model = AutoModelWithLMHead.from_pretrained('distilgpt2')    # Download model and configuration from S3 and cache.
        outputs = model.generate(max_length=40)  # do greedy decoding
        print('Generated: {}'.format(tokenizer.decode(outputs[0], skip_special_tokens=True)))

        tokenizer = AutoTokenizer.from_pretrained('openai-gpt')   # Initialize tokenizer
        model = AutoModelWithLMHead.from_pretrained('openai-gpt')    # Download model and configuration from S3 and cache.
        input_context = 'The dog'
        input_ids = tokenizer.encode(input_context, return_tensors='pt')  # encode input context
        outputs = model.generate(input_ids=input_ids, num_beams=5, num_return_sequences=3, temperature=1.5)  # generate 3 independent sequences using beam search decoding (5 beams) with sampling from initial context 'The dog'
        for i in range(3): #  3 output sequences were generated
            print('Generated {}: {}'.format(i, tokenizer.decode(outputs[i], skip_special_tokens=True)))

        tokenizer = AutoTokenizer.from_pretrained('distilgpt2')   # Initialize tokenizer
        model = AutoModelWithLMHead.from_pretrained('distilgpt2')    # Download model and configuration from S3 and cache.
        input_context = 'The dog'
        input_ids = tokenizer.encode(input_context, return_tensors='pt')  # encode input context
        outputs = model.generate(input_ids=input_ids, max_length=40, temperature=0.7, num_return_sequences=3)  # 3 generate sequences using by sampling
        for i in range(3): #  3 output sequences were generated
            print('Generated {}: {}'.format(i, tokenizer.decode(outputs[i], skip_special_tokens=True)))

        tokenizer = AutoTokenizer.from_pretrained('ctrl')   # Initialize tokenizer
        model = AutoModelWithLMHead.from_pretrained('ctrl')    # Download model and configuration from S3 and cache.
        input_context = 'Legal My neighbor is'  # "Legal" is one of the control codes for ctrl
        input_ids = tokenizer.encode(input_context, return_tensors='pt')  # encode input context
        outputs = model.generate(input_ids=input_ids, max_length=50, temperature=0.7, repetition_penalty=1.2)  # generate sequences
        print('Generated: {}'.format(tokenizer.decode(outputs[0], skip_special_tokens=True)))

        tokenizer = AutoTokenizer.from_pretrained('gpt2')   # Initialize tokenizer
        model = AutoModelWithLMHead.from_pretrained('gpt2')    # Download model and configuration from S3 and cache.
        input_context = 'My cute dog'  # "Legal" is one of the control codes for ctrl
        bad_words_ids = [tokenizer.encode(bad_word, add_prefix_space=True) for bad_word in ['idiot', 'stupid', 'shut up']]
        input_ids = tokenizer.encode(input_context, return_tensors='pt')  # encode input context
        outputs = model.generate(input_ids=input_ids, max_length=100, do_sample=True, bad_words_ids=bad_words_ids)  # generate sequences without allowing bad_words to be generated
    """

    # We cannot generate if the model does not have a LM head
    #if self.get_output_embeddings() is None:
    #    raise AttributeError(
    #        "You tried to generate sequences with a model that does not have a LM Head."
    #        "Please use another model class (e.g. `OpenAIGPTLMHeadModel`, `XLNetLMHeadModel`, `GPT2LMHeadModel`, `CTRLLMHeadModel`, `T5WithLMHeadModel`, `TransfoXLLMHeadModel`, `XLMWithLMHeadModel`, `BartForConditionalGeneration` )"
    #    )

    max_length = max_length if max_length is not None else self.config.max_length
    min_length = min_length if min_length is not None else self.config.min_length
    do_sample = do_sample if do_sample is not None else self.config.do_sample
    early_stopping = early_stopping if early_stopping is not None else self.config.early_stopping
    use_cache = use_cache if use_cache is not None else self.config.use_cache
    num_beams = num_beams if num_beams is not None else self.config.num_beams
    temperature = temperature if temperature is not None else self.config.temperature
    top_k = top_k if top_k is not None else self.config.top_k
    top_p = top_p if top_p is not None else self.config.top_p
    repetition_penalty = repetition_penalty if repetition_penalty is not None else self.config.repetition_penalty
    bos_token_id = bos_token_id if bos_token_id is not None else self.config.bos_token_id
    pad_token_id = pad_token_id if pad_token_id is not None else self.config.pad_token_id
    eos_token_id = eos_token_id if eos_token_id is not None else self.config.eos_token_id
    length_penalty = length_penalty if length_penalty is not None else self.config.length_penalty
    no_repeat_ngram_size = (
        no_repeat_ngram_size if no_repeat_ngram_size is not None else self.config.no_repeat_ngram_size
    )
    bad_words_ids = bad_words_ids if bad_words_ids is not None else self.config.bad_words_ids
    num_return_sequences = (
        num_return_sequences if num_return_sequences is not None else self.config.num_return_sequences
    )
    decoder_start_token_id = (
        decoder_start_token_id if decoder_start_token_id is not None else self.config.decoder_start_token_id
    )

    if input_ids is not None:
        batch_size = input_ids.shape[0]  # overriden by the input batch_size
    else:
        batch_size = 1

    assert isinstance(max_length, int) and max_length > 0, "`max_length` should be a strictly positive integer."
    assert isinstance(min_length, int) and min_length >= 0, "`min_length` should be a positive integer."
    assert isinstance(do_sample, bool), "`do_sample` should be a boolean."
    assert isinstance(early_stopping, bool), "`early_stopping` should be a boolean."
    assert isinstance(use_cache, bool), "`use_cache` should be a boolean."
    assert isinstance(num_beams, int) and num_beams > 0, "`num_beams` should be a strictly positive integer."
    assert temperature > 0, "`temperature` should be strictly positive."
    assert isinstance(top_k, int) and top_k >= 0, "`top_k` should be a positive integer."
    assert 0 <= top_p <= 1, "`top_p` should be between 0 and 1."
    assert repetition_penalty >= 1.0, "`repetition_penalty` should be >= 1."
    assert input_ids is not None or (
            isinstance(bos_token_id, int) and bos_token_id >= 0
    ), "If input_ids is not defined, `bos_token_id` should be a positive integer."
    assert pad_token_id is None or (
            isinstance(pad_token_id, int) and (pad_token_id >= 0)
    ), "`pad_token_id` should be a positive integer."
    assert (eos_token_id is None) or (
            isinstance(eos_token_id, int) and (eos_token_id >= 0)
    ), "`eos_token_id` should be a positive integer."
    assert length_penalty > 0, "`length_penalty` should be strictly positive."
    assert (
            isinstance(no_repeat_ngram_size, int) and no_repeat_ngram_size >= 0
    ), "`no_repeat_ngram_size` should be a positive integer."
    assert (
            isinstance(num_return_sequences, int) and num_return_sequences > 0
    ), "`num_return_sequences` should be a strictly positive integer."
    assert (
            bad_words_ids is None or isinstance(bad_words_ids, list) and isinstance(bad_words_ids[0], list)
    ), "`bad_words_ids` is either `None` or a list of lists of tokens that should not be generated"

    if input_ids is None:
        assert isinstance(bos_token_id, int) and bos_token_id >= 0, (
            "you should either supply a context to complete as `input_ids` input "
            "or a `bos_token_id` (integer >= 0) as a first token to start the generation."
        )
        input_ids = torch.full(
            (batch_size, 1), bos_token_id, dtype=torch.long, device=next(self.parameters()).device,
        )
    else:
        assert input_ids.dim() == 2, "Input prompt should be of shape (batch_size, sequence length)."

    # not allow to duplicate outputs when greedy decoding
    if do_sample is False:
        if num_beams == 1:
            # no_beam_search greedy generation conditions
            assert (
                    num_return_sequences == 1
            ), "Greedy decoding will always produce the same output for num_beams == 1 and num_return_sequences > 1. Please set num_return_sequences = 1"

        else:
            # beam_search greedy generation conditions
            assert (
                    num_beams >= num_return_sequences
            ), "Greedy beam search decoding cannot return more sequences than it has beams. Please set num_beams >= num_return_sequences"

    # create attention mask if necessary
    if (attention_mask is None) and (pad_token_id is not None) and (pad_token_id in input_ids):
        attention_mask = input_ids.ne(pad_token_id).long()
    elif attention_mask is None:
        attention_mask = input_ids.new_ones(input_ids.shape)

    # set pad_token_id to eos_token_id if not set. Important that this is done after
    # attention_mask is created
    if pad_token_id is None and eos_token_id is not None:
        logger.warning(
            "Setting `pad_token_id` to {} (first `eos_token_id`) to generate sequence".format(eos_token_id)
        )
        pad_token_id = eos_token_id

    # current position and vocab size
    if hasattr(self.config, "vocab_size"):
        vocab_size = self.config.vocab_size
    elif (
            self.config.is_encoder_decoder
            and hasattr(self.config, "decoder")
            and hasattr(self.config.decoder, "vocab_size")
    ):
        vocab_size = self.config.decoder.vocab_size

    # set effective batch size and effective batch multiplier according to do_sample
    if do_sample:
        effective_batch_size = batch_size * num_return_sequences
        effective_batch_mult = num_return_sequences
    else:
        effective_batch_size = batch_size
        effective_batch_mult = 1

    if self.config.is_encoder_decoder:
        if decoder_start_token_id is None:
            decoder_start_token_id = bos_token_id

        assert (
                decoder_start_token_id is not None
        ), "decoder_start_token_id or bos_token_id has to be defined for encoder-decoder generation"
        assert hasattr(self, "get_encoder"), "{} should have a 'get_encoder' function defined".format(self)
        assert callable(self.get_encoder), "{} should be a method".format(self.get_encoder)

        # get encoder and store encoder outputs
        encoder = self.get_encoder()

        encoder_outputs: tuple = encoder(input_ids, attention_mask=attention_mask)

    # Expand input ids if num_beams > 1 or num_return_sequences > 1
    if num_return_sequences > 1 or num_beams > 1:
        input_ids_len = input_ids.shape[-1]
        input_ids = input_ids.unsqueeze(1).expand(batch_size, effective_batch_mult * num_beams, input_ids_len)
        attention_mask = attention_mask.unsqueeze(1).expand(
            batch_size, effective_batch_mult * num_beams, input_ids_len
        )

        input_ids = input_ids.contiguous().view(
            effective_batch_size * num_beams, input_ids_len
        )  # shape: (batch_size * num_return_sequences * num_beams, cur_len)
        attention_mask = attention_mask.contiguous().view(
            effective_batch_size * num_beams, input_ids_len
        )  # shape: (batch_size * num_return_sequences * num_beams, cur_len)

    if self.config.is_encoder_decoder:
        # create empty decoder_input_ids
        input_ids = torch.full(
            (effective_batch_size * num_beams, 1),
            decoder_start_token_id,
            dtype=torch.long,
            device=next(self.parameters()).device,
        )
        cur_len = 1

        assert (
                batch_size == encoder_outputs[0].shape[0]
        ), f"expected encoder_outputs[0] to have 1st dimension bs={batch_size}, got {encoder_outputs[0].shape[0]} "

        # expand batch_idx to assign correct encoder output for expanded input_ids (due to num_beams > 1 and num_return_sequences > 1)
        expanded_batch_idxs = (
            torch.arange(batch_size)
                .view(-1, 1)
                .repeat(1, num_beams * effective_batch_mult)
                .view(-1)
                .to(input_ids.device)
        )
        # expand encoder_outputs
        encoder_outputs = (encoder_outputs[0].index_select(0, expanded_batch_idxs), *encoder_outputs[1:])

    else:
        encoder_outputs = None
        cur_len = input_ids.shape[-1]

    assert (
            cur_len < max_length
    ), f"The context has {cur_len} number of tokens, but `max_length` is only {max_length}. Please make sure that `max_length` is bigger than the number of tokens, by setting either `generate(max_length=...,...)` or `config.max_length = ...`"

    if num_beams > 1:
        output = _generate_beam_search(
            self=self,
            input_ids=input_ids,
            cur_len=cur_len,
            max_length=max_length,
            min_length=min_length,
            do_sample=do_sample,
            early_stopping=early_stopping,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            bad_words_ids=bad_words_ids,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            batch_size=effective_batch_size,
            num_return_sequences=num_return_sequences,
            length_penalty=length_penalty,
            num_beams=num_beams,
            vocab_size=vocab_size,
            encoder_outputs=encoder_outputs,
            attention_mask=attention_mask,
            use_cache=use_cache,
            look_goals=look_goals,
            key_words=key_words,
            alpha=alpha,
            model_specific_kwargs=model_specific_kwargs,
        )
    else:
        output = _generate_no_beam_search(
            self=self,
            input_ids=input_ids,
            cur_len=cur_len,
            max_length=max_length,
            min_length=min_length,
            do_sample=do_sample,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            bad_words_ids=bad_words_ids,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            batch_size=effective_batch_size,
            encoder_outputs=encoder_outputs,
            attention_mask=attention_mask,
            use_cache=use_cache,
            look_goals=look_goals,
            key_words=key_words,
            alpha=alpha,
            model_specific_kwargs=model_specific_kwargs,
        )

    return output


def _generate_no_beam_search(
        self,
        input_ids,
        cur_len,
        max_length,
        min_length,
        do_sample,
        temperature,
        top_k,
        top_p,
        repetition_penalty,
        no_repeat_ngram_size,
        bad_words_ids,
        pad_token_id,
        eos_token_id,
        batch_size,
        encoder_outputs,
        attention_mask,
        use_cache,
        look_goals,
        key_words,
        alpha,
        model_specific_kwargs,
):
    """ Generate sequences for each example without beam search (num_beams == 1).
        All returned sequence are generated independantly.
    """

    decode_kwargs = {'max_length': max_length, 'min_length': min_length, 'do_sample': do_sample,
                     'repetition_penalty': repetition_penalty, 'no_repeat_ngram_size': no_repeat_ngram_size,
                     'bad_words_ids': bad_words_ids, 'pad_token_id': pad_token_id, 'eos_token_id': eos_token_id,
                     'use_cache': use_cache, 'num_return_sequences':1, 'vocab_size':None,
                     'model_specific_kwargs': model_specific_kwargs}

    # length of generated sentences / unfinished sentences
    unfinished_sents = input_ids.new(batch_size).fill_(1)
    sent_lengths = input_ids.new(batch_size).fill_(max_length)

    #past_key_values = (encoder_outputs, None) if encoder_outputs is not None else None
    past_key_values = None

    init_len = cur_len
   
    while cur_len < max_length:
        model_inputs = self.prepare_inputs_for_generation(
            input_ids, encoder_outputs=encoder_outputs, past_key_values=past_key_values, attention_mask=attention_mask, use_cache=use_cache, **model_specific_kwargs
        )

        outputs = self(**model_inputs)
        next_token_logits = outputs[0][:, -1, :]

        scores = postprocess_next_token_scores(
            scores=next_token_logits,
            input_ids=input_ids,
            no_repeat_ngram_size=no_repeat_ngram_size,
            bad_words_ids=bad_words_ids,
            cur_len=cur_len,
            min_length=min_length,
            max_length=max_length,
            eos_token_id=eos_token_id,
            repetition_penalty=repetition_penalty,
            batch_size=batch_size,
            num_beams=1,
        )

        # if model has past, then set the past variable to speed up decoding
    
        past_key_values = outputs.past_key_values

        if do_sample:
            # Temperature (higher temperature => more likely to sample low probability tokens)
            if temperature != 1.0:
                scores = scores / temperature
            # Top-p/top-k filtering
            next_token_logscores = top_k_top_p_filtering(scores, top_k=top_k, top_p=top_p)
            # Sample
            probs = F.softmax(next_token_logscores, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).squeeze(1)
        else:

            num_cands = 5
            scores = top_k_top_p_filtering(scores, top_p=0.95)
            next_scores, next_tokens = torch.topk(scores, num_cands, dim=1, largest=True, sorted=True)
            

            ## extra key words are added as candidates
            ## token id and scores are added here.
            if key_words!=None:
                next_tokens = torch.cat((next_tokens, key_words), dim=1)
                key_words_scores = torch.gather(scores, -1, key_words)
                key_mask = key_words.eq(eos_token_id)
                key_words_scores = key_words_scores.masked_fill(key_mask, -float("inf"))
                next_scores = torch.cat((next_scores, key_words_scores), dim=1)

    
            new_batch_size = next_tokens.size(1)

            # (next_scores[0,0].float() < -0.3 )
            if (next_scores[0,0].float() < -0.5 ) and (look_goals!=None) and ((cur_len - init_len) > 2):
                for sentno in range(batch_size):
                    temp_input_ids = torch.cat([input_ids[sentno].unsqueeze(0).expand(new_batch_size, -1), next_tokens[sentno].unsqueeze(-1)], dim=-1)

                    temp_attention_mask= attention_mask[sentno].unsqueeze(0).expand(new_batch_size, -1)
                    temp_attention_mask = torch.cat([temp_attention_mask, temp_attention_mask.new_ones((temp_attention_mask.shape[0], 1))], dim=-1)

                    temp_position_ids = None
        
                    temp_look_goals = look_goals[sentno]
                    
                    """
                    if self.config.model_type=='RefinedWebModel':
                         # falcon-7b
                         temp_past = Falcon_reorder_cache(self, past_key_values, num_beams*batch_size, idxs)
                    elif self.config.model_type=="RefinedWeb":
                         # falcon-40b --lifu
                         temp_past = Falcon40b_reorder_cache(self, past_key_values, num_beams*batch_size, idxs)
                    else:
                         temp_past = self._reorder_cache(past_key_values, idxs)
                    """
                    #future_state = _evalate_self(self, temp_input_ids,  cur_len+1,
                    #                                temp_look_goals, temp_past, 2 * num_beams, temp_attention_mask,
                    #                                temp_position_ids, self._reorder_cache, **decode_kwargs)

                    ## do not use cache, change 2*num_beams to new_batch_size
                    future_state = _evalate_reverse_self(self, temp_input_ids,  init_len,
                                                   temp_look_goals, past_key_values, new_batch_size, temp_attention_mask, temp_position_ids, self._reorder_cache, **decode_kwargs)
                   

                    ## normalize score
                    future_state = future_state - torch.max(future_state)
                    
                    next_scores[sentno] += alpha*future_state

                next_scores_clone, next_scores_indices = torch.sort(next_scores, descending=True, dim=1)
                next_token = torch.gather(next_tokens, -1, next_scores_indices)
                next_token = next_token[:,0]


            else:
                # Greedy decoding
                next_token = torch.argmax(scores, dim=-1)

        # update generations and finished sentences
        if eos_token_id is not None:
            # pad finished sentences if eos_token_id exist
            tokens_to_add = next_token * unfinished_sents + (pad_token_id) * (1 - unfinished_sents)
        else:
            tokens_to_add = next_token

        # add token and increase length by one
        input_ids = torch.cat([input_ids, tokens_to_add.unsqueeze(-1)], dim=-1)
        cur_len = cur_len + 1

        if eos_token_id is not None:
            eos_in_sents = tokens_to_add == eos_token_id
            # if sentence is unfinished and the token to add is eos, sent_lengths is filled with current length
            is_sents_unfinished_and_token_to_add_is_eos = unfinished_sents.mul(eos_in_sents.long()).bool()
            sent_lengths.masked_fill_(is_sents_unfinished_and_token_to_add_is_eos, cur_len)
            # unfinished_sents is set to zero if eos in sentence
            unfinished_sents.mul_((~eos_in_sents).long())

        # stop when there is a </s> in each sentence, or if we exceed the maximul length
        if unfinished_sents.max() == 0:
            break

        # extend attention_mask for new generated input if only decoder
        if self.config.is_encoder_decoder is False:
            attention_mask = torch.cat(
                [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1
            )

    return input_ids, None, None



def _evalate_reverse_self(
        self,
        input_ids,
        init_len,
        look_goals,
        past,
        batch_size,
        attention_mask,
        position_ids,
        _reorder_cache,
        max_length,
        min_length,
        do_sample,
        repetition_penalty,
        no_repeat_ngram_size,
        bad_words_ids,
        pad_token_id,
        eos_token_id,
        num_return_sequences,
        vocab_size,
        use_cache,
        model_specific_kwargs,
):

    if look_goals.dim()==1:
        look_goals = look_goals.unsqueeze(0)
    
    total_constraint = look_goals.size(0)
    look_ahead_scores = 0
    
    for ii in range(total_constraint):
       phrase_input_ids = input_ids.new(look_goals[ii])[None, :].expand(batch_size, -1)
       phrase_input_ids = torch.cat([input_ids[:, init_len:], phrase_input_ids], dim=-1)
   
       cur_len = input_ids[:, init_len:].size(1)
       with torch.no_grad():
           follow_logits = self(input_ids=phrase_input_ids, labels=phrase_input_ids, use_cache=use_cache)[1]

       ## shift logit token < n predict n
       follow_logits = follow_logits[:, cur_len:,]
       follow_log_prob = F.log_softmax(follow_logits, dim=-1)
    
       ## skip SEP symbole
       target_l = look_goals[ii, 1:].shape[-1]
       look_ahead_score = follow_log_prob[:,torch.arange(target_l), look_goals[ii, 1:]]
       
       mask = look_goals[ii, 1:].ne(pad_token_id).float()
    
       ## add first transitioin score
       #look_ahead_scores= torch.cat((log_prob[:,-1].unsqueeze(1), look_ahead_scores), dim=-1)
       #for i in range(batch_size):
       #       look_ahead_scores.append(follow_log_prob[i]
       look_ahead_scores = look_ahead_scores +  torch.sum(look_ahead_score*mask, dim=-1)/torch.sum(mask, dim=-1)
    return look_ahead_scores/total_constraint




### evaluate constraints: compute future score
def _evalate_self(
        self,
        input_ids,
        cur_len,
        look_goals,
        past,
        batch_size,
        attention_mask,
        position_ids,
        _reorder_cache,
        max_length,
        min_length,
        do_sample,
        repetition_penalty,
        no_repeat_ngram_size,
        bad_words_ids,
        pad_token_id,
        eos_token_id,
        num_return_sequences,
        vocab_size,
        use_cache,
        model_specific_kwargs,
):
    """ Generate sequences for each example without beam search (num_beams == 1).
        All returned sequence are generated independantly.
    """
    model_inputs = self.prepare_inputs_for_generation(
            input_ids, past_key_values=past, attention_mask=attention_mask, use_cache=use_cache, **model_specific_kwargs
        )
    model_inputs["attention_mask"] = attention_mask

    outputs = self(**model_inputs)
    next_token_logits = outputs[0][:, -1, :]
    log_prob = F.log_softmax(next_token_logits, dim=-1)
    # --lifu
    past = outputs.past_key_values


    phrase_input_ids = input_ids.new(look_goals)[None, :].expand(batch_size, -1)

    #phrase_position_ids = torch.cat([position_ids[:, -1] + 1 + i for i in range(len(look_goals))], dim=-1)
    phrase_attention_mask = torch.cat(
                        [attention_mask] + [attention_mask.new_ones((attention_mask.shape[0], 1))
                                            for _ in range(len(look_goals))], dim=-1
                    )
    follow_logits = self(input_ids=phrase_input_ids, past_key_values=past,
                         attention_mask=phrase_attention_mask,
                         labels=phrase_input_ids, use_cache=use_cache)[1]



    follow_log_prob = F.log_softmax(follow_logits, dim=-1)

    target_l = look_goals.shape[-1]
    look_ahead_scores = follow_log_prob[:,torch.arange(target_l), look_goals]


    ## add first transitioin score
    look_ahead_scores= torch.cat((log_prob[:,-1].unsqueeze(1), look_ahead_scores), dim=-1)
    #for i in range(batch_size):
    #       look_ahead_scores.append(follow_log_prob[i]
    look_ahead_scores = torch.mean(look_ahead_scores, dim=-1)
    #print(look_ahead_scores)
    return look_ahead_scores



### evaluate multiple constraints: compute future score
def _evalate_self_m(
        self,
        input_ids,
        cur_len,
        look_goals,
        past,
        batch_size,
        attention_mask,
        position_ids,
        _reorder_cache,
        max_length,
        min_length,
        do_sample,
        repetition_penalty,
        no_repeat_ngram_size,
        bad_words_ids,
        pad_token_id,
        eos_token_id,
        num_return_sequences,
        vocab_size,
        use_cache,
        model_specific_kwargs,
):
    """ Generate sequences for each example without beam search (num_beams == 1).
        All returned sequence are generated independantly.
    """
    model_inputs = self.prepare_inputs_for_generation(
            input_ids, past_key_values=past, attention_mask=attention_mask, use_cache=use_cache, **model_specific_kwargs
        )
    model_inputs["attention_mask"] = attention_mask

    outputs = self(**model_inputs)
    next_token_logits = outputs[0][:, -1, :]
    log_prob = F.log_softmax(next_token_logits, dim=-1)
    # --lifu
    past = outputs.past_key_values


    phrase_input_ids = input_ids.new(look_goals)[None, :].expand(batch_size, -1)

    #phrase_position_ids = torch.cat([position_ids[:, -1] + 1 + i for i in range(len(look_goals))], dim=-1)
    phrase_attention_mask = torch.cat(
                        [attention_mask] + [attention_mask.new_ones((attention_mask.shape[0], 1))
                                            for _ in range(len(look_goals))], dim=-1
                    )
    #print(phrase_input_ids.size(), phrase_attention_mask.size())
    follow_logits = self(input_ids=phrase_input_ids, past_key_values=past,
                         attention_mask=phrase_attention_mask,
                         labels=phrase_input_ids, use_cache=use_cache)[1]



    follow_log_prob = F.log_softmax(follow_logits, dim=-1)

    target_l = look_goals.shape[-1]
    look_ahead_scores = follow_log_prob[:,torch.arange(target_l), look_goals]


    ## add first transitioin score
    look_ahead_scores= torch.cat((log_prob[:,-1].unsqueeze(1), look_ahead_scores), dim=-1)
    #for i in range(batch_size):
    #       look_ahead_scores.append(follow_log_prob[i]
    look_ahead_scores = torch.mean(look_ahead_scores, dim=-1)
    #print(look_ahead_scores)
    return look_ahead_scores



def _generate_beam_search(
        self,
        input_ids,
        cur_len,
        max_length,
        min_length,
        do_sample,
        early_stopping,
        temperature,
        top_k,
        top_p,
        repetition_penalty,
        no_repeat_ngram_size,
        bad_words_ids,
        pad_token_id,
        eos_token_id,
        batch_size,
        num_return_sequences,
        length_penalty,
        num_beams,
        vocab_size,
        encoder_outputs,
        attention_mask,
        use_cache,
        look_goals,
        key_words,
        alpha,
        model_specific_kwargs,
):
    """ Generate sequences for each example with beam search.
    """


    decode_kwargs = {'max_length': max_length, 'min_length': min_length, 'do_sample': do_sample,
                     'repetition_penalty': repetition_penalty, 'no_repeat_ngram_size': no_repeat_ngram_size,
                     'bad_words_ids': bad_words_ids, 'pad_token_id': pad_token_id, 'eos_token_id': eos_token_id,
                     'num_return_sequences': num_return_sequences, 'vocab_size': vocab_size, 'use_cache': use_cache,
                     'model_specific_kwargs': model_specific_kwargs}

    # generated hypotheses
    generated_hyps = [
        BeamHypotheses(num_beams, max_length, length_penalty, early_stopping=early_stopping)
        for _ in range(batch_size)
    ]

    # scores for each sentence in the beam
    beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=input_ids.device)

    # for greedy decoding it is made sure that only tokens of the first beam are considered to avoid sampling the exact same tokens three times
    if do_sample is False:
        beam_scores[:, 1:] = -1e9
    beam_scores = beam_scores.view(-1)  # shape (batch_size * num_beams,)

    # cache compute states
    past = (encoder_outputs, None) if encoder_outputs is not None else None
    
    # add --lifu

    init_len = cur_len
    past_key_values = None
    # done sentences
    done = [False for _ in range(batch_size)]

    use_cache=True
    while cur_len < max_length:

        ## added --Lifu
        # decoder_input_ids, past_key_values,
        # past = encoder_outputs, decoder_cached_states 
        # decoder_cached_states is removed in new transformers
        #encoder_outputs, past_key_values = past
        model_inputs = self.prepare_inputs_for_generation(
            input_ids, encoder_outputs=encoder_outputs ,past_key_values=past_key_values, attention_mask=attention_mask, use_cache=use_cache, **model_specific_kwargs
        )
        #print(model_inputs['input_ids'])
        #print( model_inputs["decoder_input_ids"].size())
        outputs = self(**model_inputs)  # (batch_size * num_beams, cur_len, vocab_size)
        next_token_logits = outputs[0][:, -1, :]  # (batch_size * num_beams, vocab_size)

        ####  remove by lifu, update 
        # if model has past, then set the past variable to speed up decoding
        #if self._use_cache(outputs, use_cache):
        #     past=outputs[1]
        

        past_key_values = outputs.past_key_values
        #print('first step past_key_values', past_key_values) 
        ## remove max_length
        if self.config.is_encoder_decoder and do_sample is False:
            next_token_logits = self.adjust_logits_during_generation(
                next_token_logits, cur_len=cur_len
            )

        scores = F.log_softmax(next_token_logits, dim=-1)  # (batch_size * num_beams, vocab_size)

        # udpate --lifu
        scores = postprocess_next_token_scores(
            scores=scores,
            input_ids=input_ids,
            no_repeat_ngram_size=no_repeat_ngram_size,
            bad_words_ids=bad_words_ids,
            cur_len=cur_len,
            min_length=min_length,
            max_length=max_length,
            eos_token_id=eos_token_id,
            repetition_penalty=repetition_penalty,
            batch_size=batch_size,
            num_beams=num_beams,
        )

        assert scores.shape == (batch_size * num_beams, vocab_size), "Shapes of scores: {} != {}".format(
            scores.shape, (batch_size * num_beams, vocab_size)
        )

        if do_sample:
            _scores = scores + beam_scores[:, None].expand_as(scores)  # (batch_size * num_beams, vocab_size)
            # Temperature
            if temperature != 1.0:
                _scores = _scores / temperature
            # Top-p/top-k filtering
            _scores = top_k_top_p_filtering(
                _scores, top_k=top_k, top_p=top_p, min_tokens_to_keep=2
            )  # (batch_size * num_beams, vocab_size)
            # re-organize to group the beam together to sample from all beam_idxs
            _scores = _scores.contiguous().view(
                batch_size, num_beams * vocab_size
            )  # (batch_size, num_beams * vocab_size)

            # Sample 2 next tokens for each beam (so we have some spare tokens and match output of greedy beam search)
            probs = F.softmax(_scores, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=2 * num_beams)  # (batch_size, num_beams * 2)
            # Compute next scores

            ### add key words into candidate pool
            #if key_words!=None:
            #    next_tokens = torch.cat( (next_tokens, key_words), dim=1)
                
            next_scores = torch.gather(_scores, -1, next_tokens)  # (batch_size, num_beams * 2)
            #if (look_goals!=None) and ((cur_len - init_len) > 2):
            #    for sentno in range(batch_size):

            # sort the sampled vector to make sure that the first num_beams samples are the best
            next_scores, next_scores_indices = torch.sort(next_scores, descending=True, dim=1)
            next_tokens = torch.gather(next_tokens, -1, next_scores_indices)  # (batch_size, num_beams * 2)

    

        else:
            
            next_scores = scores + beam_scores[:, None].expand_as(scores)  # (batch_size * num_beams, vocab_size)
            
            # re-organize to group the beam together (we are keeping top hypothesis accross beams)
            next_scores = next_scores.view(
                batch_size, num_beams * vocab_size
            )  # (batch_size, num_beams * vocab_size)
            
            ## consider add key words in the first beam
            ## could extend to all beams in the future  --lifu
            first_beam_next_score = next_scores[:, :vocab_size].clone().detach()
            
            next_scores, next_tokens = torch.topk(next_scores, 2 * num_beams, dim=1, largest=True, sorted=True)
            
            ## extra key words are added as candidates
            ## token id and scores are added here.
            if key_words!=None:
                next_tokens = torch.cat((next_tokens, key_words), dim=1)
                key_words_scores = torch.gather(first_beam_next_score, -1, key_words)
                key_mask = key_words.eq(eos_token_id)
                key_words_scores = key_words_scores.masked_fill(key_mask, -float("inf"))
                next_scores = torch.cat((next_scores, key_words_scores), dim=1)
            
            beam_id = next_tokens // vocab_size  # (batch_size, num_beams * 2)
            token_id = next_tokens % vocab_size  # (batch_size, num_beams * 2)
   
            new_batch_size = next_tokens.size(1)
            #print('next_tokens', next_tokens.size())

            next_scores_clone = next_scores.clone().detach()
            if (look_goals!=None) and ((cur_len - init_len) > 2):
                #next_scores_clone = next_scores.clone().detach()
                #print(input_ids[:num_beams, :])
                for sentno in range(batch_size):

                    #tmprows  = slice(sentno *2* num_beams, sentno * 2*num_beams +2* num_beams)
                    rows = beam_id[sentno] + sentno*num_beams
                    #print(rows)

                    tt_input = token_id[sentno]
                    #print('canid pool size', tt_input.size())
                    ## extra key words are added as candidates
                

                    #print(tmp_beam_id)
                    ## 10 original
                    #prun_size = 2
                    #tmp_scores = -100000*torch.ones((num_beams, vocab_size), dtype=torch.float, device=input_ids.device)
                   
                    #rows = slice(sentno * num_beams, sentno * num_beams + num_beams)

                    #idxs = torch.arange(sentno * num_beams, sentno * num_beams + num_beams, device=past_key_values[0][0].device)
                    #idxs = rows.to(past_key_values[0][0].device)
                    idxs = rows.to('cuda')
                    #print(batch_size, num_beams, input_ids.size(), next_tokens.size(), input_ids[rows].size(), next_tokens[rows][rows].size()) 
                    #_, tmp_next_tokens = torch.topk(scores[rows], prun_size, dim=1, largest=True, sorted=True)
                    #print(tmp_next_tokens.size())
                    #print([tokenizer.decode(ii) for ii in tmp_next_tokens.view(-1)])
                    #tt_input = input_ids[rows]
                    #print(tt_input.size())
                    #print([tokenizer.decode(tt_input[i]) for i in range(tt_input.shape[0])])

                    ### start to score next tokens
                    temp_input_ids = torch.cat([input_ids[rows], tt_input.unsqueeze(1)], dim=-1)
                    temp_attention_mask= attention_mask[rows]
                    temp_attention_mask = temp_attention_mask if self.config.is_encoder_decoder else \
                             torch.cat([temp_attention_mask, temp_attention_mask.new_ones((temp_attention_mask.shape[0], 1))], dim=-1)


                    #temp_input_ids = torch.cat([input_ids[rows].unsqueeze(1).expand(num_beams,prun_size, input_ids.shape[-1]).contiguous().view(num_beams*prun_size, input_ids.shape[-1]), tmp_next_tokens.view(-1).unsqueeze(1)],dim=-1)
                    #temp_input_ids = torch.cat([input_ids[rows], tt_input], dim=-1)

                    #temp_attention_mask= attention_mask[rows].unsqueeze(1).expand(num_beams,prun_size, input_ids.shape[-1]).contiguous().view(num_beams*prun_size, input_ids.shape[-1])
                    #temp_attention_mask = temp_attention_mask if self.config.is_encoder_decoder else \
                    #         torch.cat([temp_attention_mask, temp_attention_mask.new_ones((temp_attention_mask.shape[0], 1))], dim=-1)
                             
                    #temp_position_ids = position_ids[rows].unsqueeze(1).expand(num_beams,prun_size, input_ids.shape[-1]).contiguous().view(num_beams*prun_size, input_ids.shape[-1])
                    temp_position_ids = None
                    temp_look_goals = look_goals[sentno]

                    #beam_idx = input_ids.new([i for i in range(num_beams) for _ in range(prun_size)])
                    #idxs = idxs.unsqueeze(1).expand(num_beams,prun_size).contiguous().view(-1)

                    if self.config.model_type=='RefinedWebModel':
                         # falcon-7b
                         temp_past = Falcon_reorder_cache(self, past_key_values, num_beams*batch_size, idxs)
                    elif self.config.model_type=="RefinedWeb":
                         # falcon-40b --lifu
                         temp_past = Falcon40b_reorder_cache(self, past_key_values, num_beams*batch_size, idxs)
                    else:
                         temp_past = self._reorder_cache(past_key_values, idxs)
                
                    #future_state = _evalate_self(self, temp_input_ids,  cur_len+1,
                    #                                temp_look_goals, temp_past, 2 * num_beams, temp_attention_mask,
                    #                                temp_position_ids, self._reorder_cache, **decode_kwargs)
                    
                    ## do not use cache, change 2*num_beams to new_batch_size
                    future_state = _evalate_reverse_self(self, temp_input_ids,  init_len,
                                                   temp_look_goals, temp_past, new_batch_size, temp_attention_mask,
                                                    temp_position_ids, self._reorder_cache, **decode_kwargs)

        
                    ## normalize score
            
                    future_state = future_state - torch.max(future_state)
                    #next_scores_clone[sentno] += alpha*math.tanh((cur_len - init_len )/5.0)*future_state
                    ## here multiply length becase the beam score is also related to length
                    score_thres_mask = torch.le(next_scores_clone[sentno], np.log(0.5))

                    ## next_scores_clone[sentno] += alpha*future_state #

                    # if next token prob is very high, it should prefer to use it
                    next_scores_clone[sentno] += alpha*future_state*score_thres_mask

                    #next_scores_clone[sentno] += alpha*future_state*(cur_len - init_len)/5

                    #print(future_state)
                    #future_state =future_state.contiguous().view(num_beams,prun_size)
                    #for i in range(num_beams):
                    #    tmp_scores[i, tmp_next_tokens[i]] = future_state[i]

                    #print(tmp_scores[torch.arange(num_beams), tmp_next_tokens].size())
                    #= future_state 
                    #print(future_state)
                    #next_scores[rows] = tmp_scores
                    #print(cur_len)
                    #
                    #scores_clone[rows] += alpha*math.tanh((cur_len-init_len)/10.0)*tmp_scores 
   
                next_scores_clone, next_scores_indices = torch.sort(next_scores_clone, descending=True, dim=1)
                next_tokens = torch.gather(next_tokens, -1, next_scores_indices)

            if key_words!=None:
                next_scores_clone= next_scores_clone[:,:2 * num_beams]
                next_tokens = next_tokens[:,:2 * num_beams]
                next_scores= next_scores[:,:2 * num_beams]
        ### ****************************************************
        ## after finalized , the constrained score is also used for ranking --lifu
        #### 
        #next_scores_clone, _  = torch.sort(next_scores_clone, descending=True, dim=1)
    

        assert next_scores.size() == next_tokens.size() == (batch_size, 2 * num_beams)

        # next batch beam content
        next_batch_beam = []

        # for each sentence
        for batch_idx in range(batch_size):

            # if we are done with this sentence, add a pad token
            if done[batch_idx]:
                assert (
                        len(generated_hyps[batch_idx]) >= num_beams
                ), "Batch can only be done if at least {} beams have been generated".format(num_beams)
                assert (
                        eos_token_id is not None and pad_token_id is not None
                ), "generated beams >= num_beams -> eos_token_id and pad_token have to be defined"
                next_batch_beam.extend([(0, pad_token_id, 0)] * num_beams)  # pad the batch
                continue

            # next sentence beam content, this will get added to next_batch_beam
            next_sent_beam = []

            # next tokens for this sentence, add constraint score if end of sentence   -----Lifu
            for beam_token_rank, (beam_token_id, beam_token_score, beam_token_score_clone) in enumerate(
                    zip(next_tokens[batch_idx], next_scores[batch_idx], next_scores_clone[batch_idx])
            ):
                # get beam and token IDs
                beam_id = beam_token_id // vocab_size
                token_id = beam_token_id % vocab_size

                effective_beam_id = batch_idx * num_beams + beam_id
                # add to generated hypotheses if end of sentence
                if (eos_token_id is not None) and (token_id.item() == eos_token_id):
                    # if beam_token does not belong to top num_beams tokens, it should not be added
                    is_beam_token_worse_than_top_num_beams = beam_token_rank >= num_beams
                    if is_beam_token_worse_than_top_num_beams:
                        continue
                    generated_hyps[batch_idx].add(
                        input_ids[effective_beam_id].clone(), beam_token_score_clone.item(),
                    )
                    ## changed above --lifu
                else:
                    # add next predicted token since it is not eos_token
                    next_sent_beam.append((beam_token_score, token_id, effective_beam_id))

                # once the beam for next step is full, don't add more tokens to it.
                if len(next_sent_beam) == num_beams:
                    break

            # Check if we are done so that we can save a pad step if all(done)
            done[batch_idx] = done[batch_idx] or generated_hyps[batch_idx].is_done(
                next_scores[batch_idx].max().item(), cur_len
            )

            # update next beam content
            assert len(next_sent_beam) == num_beams, "Beam should always be full"
            next_batch_beam.extend(next_sent_beam)
            assert len(next_batch_beam) == num_beams * (batch_idx + 1), "We should have added num_beams each step"

        # stop when we are done with each sentence
        if all(done):
            break

        # sanity check / prepare next batch
        assert len(next_batch_beam) == batch_size * num_beams
        beam_scores = beam_scores.new([x[0] for x in next_batch_beam])
        beam_tokens = input_ids.new([x[1] for x in next_batch_beam])
        beam_idx = input_ids.new([x[2] for x in next_batch_beam])

        # re-order batch and update current length
        input_ids = input_ids[beam_idx, :]
        input_ids = torch.cat([input_ids, beam_tokens.unsqueeze(1)], dim=-1)
        cur_len = cur_len + 1

        # update --lifu
        # re-order internal states
        if past_key_values is not None:
            #print(past)
            past_key_values = self._reorder_cache(past_key_values, beam_idx)

        # extend attention_mask for new generated input if only decoder
        if self.config.is_encoder_decoder is False:
            attention_mask = torch.cat(
                [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1
            )

    # finalize all open beam hypotheses and add to generated hypotheses
    for batch_idx in range(batch_size):
        if done[batch_idx]:
            continue

        # test that beam scores match previously calculated scores if not eos and batch_idx not done
        if eos_token_id is not None and all(
                (token_id % vocab_size).item() != eos_token_id for token_id in next_tokens[batch_idx]
        ):
            assert torch.all(
                next_scores[batch_idx, :num_beams] == beam_scores.view(batch_size, num_beams)[batch_idx]
            ), "If batch_idx is not done, final next scores: {} have to equal to accumulated beam_scores: {}".format(
                next_scores[:, :num_beams][batch_idx], beam_scores.view(batch_size, num_beams)[batch_idx],
            )

        # need to add best num_beams hypotheses to generated hyps
        for beam_id in range(num_beams):
            effective_beam_id = batch_idx * num_beams + beam_id
            final_score = beam_scores[effective_beam_id].item()
            final_tokens = input_ids[effective_beam_id]
            generated_hyps[batch_idx].add(final_tokens, final_score)

    # depending on whether greedy generation is wanted or not define different output_batch_size and output_num_return_sequences_per_batch
    output_batch_size = batch_size if do_sample else batch_size * num_return_sequences
    output_num_return_sequences_per_batch = 1 if do_sample else num_return_sequences

    # select the best hypotheses
    sent_lengths = input_ids.new(output_batch_size)
    best, best_scores, best_sum_logprobs = [], [], []

    # retrieve best hypotheses
    for i, hypotheses in enumerate(generated_hyps):
        sorted_hyps = sorted(hypotheses.beams, key=lambda x: x[0])
        for j in range(output_num_return_sequences_per_batch):
            effective_batch_idx = output_num_return_sequences_per_batch * i + j
            best_score, best_hyp = sorted_hyps.pop()
            sent_lengths[effective_batch_idx] = len(best_hyp)
            best.append(best_hyp)
            best_scores.append(best_score)
            best_sum_logprobs.append(best_score * (len(best_hyp) ** length_penalty))

    # shorter batches are padded
    if sent_lengths.min().item() != sent_lengths.max().item():
        assert pad_token_id is not None, "`Pad_token_id` has to be defined"
        sent_max_len = min(sent_lengths.max().item() + 1, max_length)
        decoded = input_ids.new(output_batch_size, sent_max_len).fill_(pad_token_id)

        # fill with hypothesis and eos_token_id if necessary
        for i, hypo in enumerate(best):
            decoded[i, : sent_lengths[i]] = hypo
            if sent_lengths[i] < max_length:
                decoded[i, sent_lengths[i]] = eos_token_id
    else:
        # none of the hypotheses have an eos_token
        assert (len(hypo) == max_length for hypo in best)
        decoded = torch.stack(best).type(torch.long).to(next(self.parameters()).device)

    return decoded, best_scores, best_sum_logprobs



def top_k_top_p_filtering(
    logits,
    top_k = 0,
    top_p = 1.0,
    filter_value  = -float("Inf"),
    min_tokens_to_keep = 1,
):
    """Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
    Args:
        logits: logits distribution shape (batch size, vocabulary size)
        if top_k > 0: keep only top k tokens with highest probability (top-k filtering).
        if top_p < 1.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
            Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        Make sure we keep at least min_tokens_to_keep per batch example in the output
    From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    if top_k > 0:
        top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))  # Safety check
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold (token with 0 are kept)
        sorted_indices_to_remove = cumulative_probs > top_p
        if min_tokens_to_keep > 1:
            # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
            sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value
    return logits



class BeamHypotheses(object):
    def __init__(self, num_beams, max_length, length_penalty, early_stopping):
        """
        Initialize n-best list of hypotheses.
        """
        self.max_length = max_length - 1  # ignoring bos_token
        self.length_penalty = length_penalty
        self.early_stopping = early_stopping
        self.num_beams = num_beams
        self.beams = []
        self.worst_score = 1e9

    def __len__(self):
        """
        Number of hypotheses in the list.
        """
        return len(self.beams)

    def add(self, hyp, sum_logprobs):
        """
        Add a new hypothesis to the list.
        """
        score = sum_logprobs / len(hyp) ** self.length_penalty
        if len(self) < self.num_beams or score > self.worst_score:
            self.beams.append((score, hyp))
            if len(self) > self.num_beams:
                sorted_scores = sorted([(s, idx) for idx, (s, _) in enumerate(self.beams)])
                del self.beams[sorted_scores[0][1]]
                self.worst_score = sorted_scores[1][0]
            else:
                self.worst_score = min(score, self.worst_score)

    def is_done(self, best_sum_logprobs, cur_len):
        """
        If there are enough hypotheses and that none of the hypotheses being generated
        can become better than the worst one in the heap, then we are done with this sentence.
        """

        if len(self) < self.num_beams:
            return False
        elif self.early_stopping:
            return True
        else:
            if cur_len is None:
                cur_len = self.max_length
            cur_score = best_sum_logprobs / cur_len ** self.length_penalty
            ret = self.worst_score >= cur_score
            return ret
