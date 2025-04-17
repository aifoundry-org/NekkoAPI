#!/bin/bash

python3 -m llama_cpp.server --host $HOST --port $PORT \
  --model $MODEL_PATH \
  --model_alias $MODEL_ALIAS \
  --n_ctx $N_CTX \
  --n_batch $N_BATCH \
  --n_ubatch $N_UBATCH \
  --n_threads $N_THREADS \
  --n_threads_batch $N_THREADS_BATCH \
  --mul_mat_q $MUL_MAT_Q \
  --flash_attn $FLASH_ATTN \
  --numa $NUMA \
  --cache $CACHE \
  --cache_type $CACHE_TYPE \
  --cache_size $CACHE_SIZE
