.PHONY: perf_test_all perf_test_vllm perf_test_nekko perf_test_ollama cleanup

perf_test_all: perf_test_nekko perf_test_ollama perf_test_vllm

clone_vllm:
	@mkdir -p build
	@if [ ! -d "build/vllm" ]; then \
		git clone https://github.com/vllm-project/vllm.git build/vllm; \
	fi

clone_llmperf:
	@if [ ! -d "build/llmperf" ]; then \
		git clone https://github.com/ray-project/llmperf.git build/llmperf; \
	fi

models:
	@mkdir -p models

# ollama healthcheck depends on qwen2.5-3b-instruct-q5_k_m.gguf
models/qwen2.5-3b-instruct-q5_k_m.gguf: | models
	curl -L https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main/qwen2.5-3b-instruct-q5_k_m.gguf -o $@

ollama-setup: | models/qwen2.5-3b-instruct-q5_k_m.gguf
	@chmod +x scripts/ollama-setup.sh && ./scripts/ollama-setup.sh

perf_test_vllm: | clone_llmperf clone_vllm
	@DOCKER_GID=$(getent group docker | cut -d: -f3) \
		COMPOSE_PROFILES=vllm_mode \
		docker compose build llm_perf_vllm && \
		DOCKER_GID=$(getent group docker | cut -d: -f3) \
		COMPOSE_PROFILES=vllm_mode \
		docker compose up --exit-code-from llm_perf_vllm && \
		docker compose down

perf_test_nekko: | clone_llmperf models/qwen2.5-3b-instruct-q5_k_m.gguf
	@DOCKER_GID=$(getent group docker | cut -d: -f3) \
		COMPOSE_PROFILES=nekko_mode \
		docker compose build llm_perf_nekko && \
		DOCKER_GID=$(getent group docker | cut -d: -f3) \
		COMPOSE_PROFILES=nekko_mode \
		docker compose up --exit-code-from llm_perf_nekko && \
		docker compose down

perf_test_ollama: | clone_llmperf models/qwen2.5-3b-instruct-q5_k_m.gguf
	@DOCKER_GID=$(getent group docker | cut -d: -f3) \
		COMPOSE_PROFILES=ollama_mode \
		docker compose build llm_perf_ollama && \
		DOCKER_GID=$(getent group docker | cut -d: -f3) \
		COMPOSE_PROFILES=ollama_mode \
		docker compose up --exit-code-from llm_perf_ollama && \
		docker compose down

cleanup:
	@rm -rf build && rm -rf models && rm -rf results
