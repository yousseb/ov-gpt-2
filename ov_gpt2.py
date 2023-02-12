"""
Based on https://github.com/openvinotoolkit/open_model_zoo/blob/master/demos/gpt2_text_prediction_demo/python/gpt2_text_prediction_demo.py
     and https://github.com/openvinotoolkit/openvino_notebooks/blob/main/notebooks/223-gpt2-text-prediction/223-gpt2-text-prediction.ipynb
On Windows, you also need: https://aka.ms/vs/17/release/vc_redist.x64.exe
"""

import logging as log
import time
from pathlib import Path
import numpy as np
import warnings
from openvino.runtime import Core, get_version
from transformers import GPT2Tokenizer, GPT2LMHeadModel

eos_token_id = ""

warnings.simplefilter(action='ignore', category=FutureWarning)


class OVGPT2Config:
    model = Path('model') / Path('gpt-2.xml')
    device = 'CPU'               # 'CPU', 'GPU', 'Auto', etc..
    top_k = 20                   # Number of tokens with the highest probability which will be kept for generation
    dynamic_shapes = True        # Run with dynamic input sequence. False = input sequence will be padded to max_seq_len
    max_sequence_length = 128    # When dynamic_shapes = False, use this maximum sequence length for stop iteration


class OVGPT2:
    def __init__(self, config: OVGPT2Config):
        global eos_token_id
        self.config = config

        # create tokenizer
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        eos_token_id = self.tokenizer.eos_token_id
        log.debug('Tokenizer configured')

        log.info('OpenVINO Runtime build: {}'.format(get_version()))
        self.core = Core()

        # read model
        log.info('Reading model {}'.format(config.model))
        self.model = self.core.read_model(config.model)

        self.input_tensor = self.model.inputs[0].any_name

        # validate model
        self._validate_model()

        # prepare tensor shapes
        # self._prepare_model_shapes()
        # assign dynamic shapes to every input layer
        for input_layer in self.model.inputs:
            input_shape = input_layer.partial_shape
            input_shape[0] = -1
            input_shape[1] = -1
            self.model.reshape({input_layer: input_shape})

        # load model to the device
        self.compiled_model = self.core.compile_model(self.model, config.device)
        self.output_tensor = self.compiled_model.outputs[0]
        self.infer_request = self.compiled_model.create_infer_request()
        log.info('Model {} is loaded to {}'.format(config.model, config.device))

    def _validate_model(self):
        # check number inputs and outputs
        if len(self.model.inputs) != 2:
            raise RuntimeError('Expected model with single input, while provided {}'.format(
                len(self.model.inputs)))
        if len(self.model.outputs) != 1:
            raise RuntimeError('Expected model with single output, while provided {}'.format(
                len(self.model.outputs)))

    # this function converts text to tokens
    def tokenize(self, text):
        """
        tokenize input text using GPT2 tokenizer

        Parameters:
          text, str - input text
        Returns:
          input_ids - np.array with input token ids
          attention_mask - np.array with 0 in place, where should be padding and 1 for places where original tokens are located, represents attention mask for model
        """

        inputs = self.tokenizer(text, return_tensors="np")
        return inputs["input_ids"], inputs["attention_mask"]

    def softmax(self, x):
        e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        summation = e_x.sum(axis=-1, keepdims=True)
        return e_x / summation

    def process_logits(self, cur_length, scores, eos_token_id, min_length=0):
        """
        reduce probability for padded indicies

        Parameters:
          cur_length - current length of input sequence
          scores - model output logits
          eos_token_id - index of end of string token in model vocab
          min_length - minimum length for appling postprocessing
        """
        if cur_length < min_length:
            scores[:, eos_token_id] = -float("inf")
        return scores

    def get_top_k_logits(self, scores, top_k):
        """
        perform top-k sampling

        Parameters:
          scores - model output logits
          top_k - number of elements with highest probability to select
        """
        filter_value = -float("inf")
        top_k = min(max(top_k, 1), scores.shape[-1])
        top_k_scores = -np.sort(-scores)[:, :top_k]
        indices_to_remove = scores < np.min(top_k_scores)
        filtred_scores = np.ma.array(scores, mask=indices_to_remove,
                                     fill_value=filter_value).filled()
        return filtred_scores

    def generate_sequence(self, input_ids, attention_mask, eos_token_id=eos_token_id):
        """
        text prediction cycle.

        Parameters:
          input_ids: tokenized input ids for model
          attention_mask: attention mask for model
          eos_token_id: end of sequence index from vocab
        Returns:
          predicted token ids sequence
        """
        output_key = self.compiled_model.output(0)

        while True:
            cur_input_len = len(input_ids[0])
            if not self.config.dynamic_shapes:
                pad_len = self.config.max_sequence_length - cur_input_len
                model_input_ids = np.concatenate((input_ids, [[eos_token_id] * pad_len]), axis=-1)
                model_input_attention_mask = np.concatenate((attention_mask, [[0] * pad_len]), axis=-1)
            else:
                model_input_ids = input_ids
                model_input_attention_mask = attention_mask
            outputs = self.compiled_model({"input_ids": model_input_ids, "attention_mask": model_input_attention_mask})[
                output_key]
            next_token_logits = outputs[:, cur_input_len - 1, :]
            # pre-process distribution
            next_token_scores = self.process_logits(cur_input_len,
                                               next_token_logits, eos_token_id)
            top_k = self.config.top_k
            next_token_scores = self.get_top_k_logits(next_token_scores, top_k)
            # get next token id
            probs = self.softmax(next_token_scores)
            next_tokens = np.random.choice(probs.shape[-1], 1,
                                           p=probs[0], replace=True)
            # break the loop if max length or end of text token is reached
            if cur_input_len == self.config.max_sequence_length or next_tokens == eos_token_id:
                break
            else:
                input_ids = np.concatenate((input_ids, [next_tokens]), axis=-1)
                attention_mask = np.concatenate((attention_mask, [[1] * len(next_tokens)]), axis=-1)
        return input_ids

    def infer(self, prompt: str) -> str:
        input_ids, attention_mask = self.tokenize(prompt)

        start = time.perf_counter()
        output_ids = self.generate_sequence(input_ids, attention_mask)
        end = time.perf_counter()
        output_text = ""
        # Convert IDs to words and make the sentence from it
        for i in output_ids[0]:
            output_text += self.tokenizer.convert_tokens_to_string(self.tokenizer._convert_id_to_token(i))

        log.debug(f"OUTPUT: {output_text}")
        log.info(f"Generation took {end - start:.3f} s")
        return '{}'.format(output_text)
