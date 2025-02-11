.PHONY: perf_test perf_test_prepare download-models perf_test_build perf_test_up clean

perf_test: perf_test_prepare perf_test_build download-models perf_test_up

perf_test_prepare:
	@mkdir -p build
	@if [ ! -d "build/vllm" ]; then \
		git clone https://github.com/vllm-project/vllm.git build/vllm; \
	fi
	@if [ ! -d "build/llmperf" ]; then \
		git clone https://github.com/ray-project/llmperf.git build/llmperf; \
	fi

perf_test_build:
	@docker compose build

perf_test_up:
	@docker compose up --exit-code-from llm_perf && docker compose down

download-models: models/qwen2.5-3b-instruct-q5_k_m.gguf

models/qwen2.5-3b-instruct-q5_k_m.gguf: | models
	curl -L https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main/qwen2.5-3b-instruct-q5_k_m.gguf -o $@

models:
	mkdir models

clean:
	@rm -rf build && rm -rf models && rm -rf results
