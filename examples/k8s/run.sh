#!/bin/bash

python3 -m llama_cpp.server --host $HOST --port $PORT --model $MODEL_PATH --model_alias $MODEL_ALIAS

