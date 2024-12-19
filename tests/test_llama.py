import ctypes
import multiprocessing

import numpy as np
from scipy.special import log_softmax

from huggingface_hub import hf_hub_download

import pytest

import llama_cpp
import llama_cpp._internals as internals


MODEL = "./vendor/llama.cpp/models/ggml-vocab-llama-spm.gguf"


def test_llama_cpp_version():
    assert llama_cpp.__version__


def test_llama_cpp_tokenization():
    llama = llama_cpp.Llama(model_path=MODEL, vocab_only=True, verbose=False)

    assert llama
    assert llama._ctx.ctx is not None

    text = b"Hello World"

    tokens = llama.tokenize(text)
    assert tokens[0] == llama.token_bos()
    assert tokens == [1, 15043, 2787]
    detokenized = llama.detokenize(tokens)
    assert detokenized == text

    tokens = llama.tokenize(text, add_bos=False)
    assert tokens[0] != llama.token_bos()
    assert tokens == [15043, 2787]

    detokenized = llama.detokenize(tokens)
    assert detokenized != text

    text = b"Hello World</s>"
    tokens = llama.tokenize(text)
    assert tokens[-1] != llama.token_eos()
    assert tokens == [1, 15043, 2787, 829, 29879, 29958]

    tokens = llama.tokenize(text, special=True)
    assert tokens[-1] == llama.token_eos()
    assert tokens == [1, 15043, 2787, 2]

    text = b""
    tokens = llama.tokenize(text, add_bos=True, special=True)
    assert tokens[-1] != llama.token_eos()
    assert tokens == [llama.token_bos()]
    assert text == llama.detokenize(tokens)


@pytest.fixture
def llama_cpp_model_path():
    repo_id = "Qwen/Qwen2-0.5B-Instruct-GGUF"
    filename = "qwen2-0_5b-instruct-q8_0.gguf"
    model_path = hf_hub_download(repo_id, filename)
    return model_path


def test_real_model(llama_cpp_model_path):
    import os
    assert os.path.exists(llama_cpp_model_path)

    params = llama_cpp.llama_model_default_params()
    params.use_mmap = llama_cpp.llama_supports_mmap()
    params.use_mlock = llama_cpp.llama_supports_mlock()
    params.check_tensors = False

    model = internals.LlamaModel(path_model=llama_cpp_model_path, params=params)

    cparams = llama_cpp.llama_context_default_params()
    cparams.n_ctx = 16
    cparams.n_batch = 16
    cparams.n_ubatch = 16
    cparams.n_threads = multiprocessing.cpu_count()
    cparams.n_threads_batch = multiprocessing.cpu_count()
    cparams.logits_all = False
    cparams.flash_attn = True

    context = internals.LlamaContext(model=model, params=cparams)
    tokens = model.tokenize(b"Hello, world!", add_bos=True, special=True)

    assert tokens == [9707, 11, 1879, 0]

    tokens = model.tokenize(b"The quick brown fox jumps", add_bos=True, special=True)

    batch = internals.LlamaBatch(n_tokens=len(tokens), embd=0, n_seq_max=1)

    seed = 1337
    sampler = internals.LlamaSampler()
    sampler.add_top_k(50)
    sampler.add_top_p(0.9, 1)
    sampler.add_temp(0.8)
    sampler.add_dist(seed)

    result = tokens
    n_eval = 0
    for _ in range(4):
        batch.set_batch(tokens, n_past=n_eval, logits_all=False)
        context.decode(batch)
        n_eval += len(tokens)
        token_id = sampler.sample(context, -1)
        tokens = [token_id]
        result += tokens

    output = result[5:]
    output_text = model.detokenize(output, special=True)
    assert output_text == b" over the lazy dog"

def build_model_with_params(path, **kwargs):
    return llama_cpp.Llama(
        path,
        n_ctx=32,
        n_batch=32,
        n_ubatch=32,
        n_threads=multiprocessing.cpu_count(),
        n_threads_batch=multiprocessing.cpu_count(),
        **kwargs
    )

def build_output_from_model(
    model: llama_cpp.Llama,
    prompt: str,
    **kwargs
):
    return model.create_completion(
        prompt=prompt,
        max_tokens=4,
        top_k=50,
        top_p=0.9,
        temperature=0.8,
        **kwargs
    )

def test_presence_penalty(llama_cpp_model_path):
    model = build_model_with_params(
        llama_cpp_model_path,
        logits_all=False,
        flash_attn=True,
    )

    output = build_output_from_model(
        model=model,
        prompt="The quick brown fox jumps",
        presence_penalty=2.0
    )

    assert output is not None

def test_real_llama(llama_cpp_model_path):
    model = build_model_with_params(
        llama_cpp_model_path,
        logits_all=False,
        flash_attn=True,
    )

    assert build_output_from_model(
        model,
        "The quick brown fox jumps",
        seed=1337,
    )["choices"][0]["text"] == " over the lazy dog"

    assert build_output_from_model(
        model,
        "The capital of france is paris, 'true' or 'false'?:\n",
        seed=1337,
        grammar=llama_cpp.LlamaGrammar.from_string(
            'root ::= "true" | "false"'
        ))["choices"][0]["text"] == "true"

    suffix = b"rot"
    tokens = model.tokenize(suffix, add_bos=True, special=True)
    def logit_processor_func(input_ids, logits):
        for token in tokens:
            logits[token] *= 1000
        return logits

    logit_processors = llama_cpp.LogitsProcessorList(
        [logit_processor_func]
    )

    assert build_output_from_model(
        model,
        "The capital of france is par",
        seed=1337,
        logits_processor=logit_processors
    )["choices"][0]["text"].lower().startswith("rot")

    model.set_seed(1337)

    grammar_string = "root ::= " + ' | '.join(list(map(lambda x : f'"{x}"', range(1, 11))))

    state = model.save_state()
    grammar = grammar=llama_cpp.LlamaGrammar.from_string(grammar_string)
    prompt = "Pick a number from 1 to 10?:\n"

    number_1 = build_output_from_model(
        model,
        prompt,
        grammar=grammar
    )["choices"][0]["text"]

    number_2 = build_output_from_model(
        model,
        prompt,
        grammar=grammar
    )["choices"][0]["text"]

    model.load_state(state)

    number_3 = build_output_from_model(
        model,
        prompt,
        grammar=grammar
    )["choices"][0]["text"]

    assert number_1 != number_2
    assert number_1 == number_3
