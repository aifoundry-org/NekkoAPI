update:
	poetry install
	git submodule update --init --recursive

update.vendor:
	cd vendor/llama.cpp && git pull origin master

deps:
	python3 -m pip install --upgrade pip
	python3 -m pip install -e ".[all]"

build:
	python3 -m pip install --verbose -e .

build.debug:
	python3 -m pip install \
		--verbose \
		--config-settings=cmake.verbose=true \
		--config-settings=logging.level=INFO \
		--config-settings=install.strip=false  \
		--config-settings=cmake.args="-DCMAKE_BUILD_TYPE=Debug;-DCMAKE_C_FLAGS='-ggdb -O0';-DCMAKE_CXX_FLAGS='-ggdb -O0'" \
		--editable .

build.debug.extra:
	python3 -m pip install \
		--verbose \
		--config-settings=cmake.verbose=true \
		--config-settings=logging.level=INFO \
		--config-settings=install.strip=false  \
		--config-settings=cmake.args="-DCMAKE_BUILD_TYPE=Debug;-DCMAKE_C_FLAGS='-fsanitize=address -ggdb -O0';-DCMAKE_CXX_FLAGS='-fsanitize=address -ggdb -O0'" \
		--editable .

build.cuda:
	CMAKE_ARGS="-DGGML_CUDA=on" python3 -m pip install --verbose -e .

build.openblas:
	CMAKE_ARGS="-DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS" python3 -m pip install --verbose -e .

test:
	python3 -m pytest --full-trace -v

test-docker:
	docker run --rm nekko-api:latest /bin/sh -c "python3 -m pip install -e \".[all]\" && python3 -m pytest --full-trace -v"

docker:
	docker build -t nekko-api:latest -f docker/simple/Dockerfile .

run-server:
	python3 -m llama_cpp.server --model ${MODEL}

run:
	python3 -m llama_cpp.server --config_file=./examples/settings.json

run-example-docker: example-models
	docker run \
	  -v ./models:/app/models \
	  -v ./examples/settings.json:/app/settings.json \
	  --cap-add SYS_RESOURCE \
	  -p 8000:8000 \
	  -e CONFIG_FILE=settings.json \
	  -it \
	  nekko-api

run-demo: example-models
	docker compose -f docker/web/docker-compose.yml up

example-models: models/SmolLM2-135M-Instruct-Q6_K.gguf models/Llama-3.2-1B-Instruct-Q5_K_S.gguf models/OLMo-7B-Instruct-hf-0724-Q4_K.gguf

models/SmolLM2-135M-Instruct-Q6_K.gguf: | models
	curl -L https://huggingface.co/lmstudio-community/SmolLM2-135M-Instruct-GGUF/resolve/main/SmolLM2-135M-Instruct-Q6_K.gguf -o $@

models/Llama-3.2-1B-Instruct-Q5_K_S.gguf: | models
	curl -L https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q5_K_S.gguf -o $@

models/OLMo-7B-Instruct-hf-0724-Q4_K.gguf: | models
	curl -L https://huggingface.co/aifoundry-org/OLMo-7B-0724-Instruct-hf-Quantized/resolve/main/OLMo-7B-Instruct-hf-0724-Q4_K.gguf -o $@

models:
	mkdir models

clean:
	- cd vendor/llama.cpp && make clean
	- cd vendor/llama.cpp && rm libllama.so
	- rm -rf _skbuild
	- rm llama_cpp/lib/*.so
	- rm llama_cpp/lib/*.dylib
	- rm llama_cpp/lib/*.metal
	- rm llama_cpp/lib/*.dll
	- rm llama_cpp/lib/*.lib
	- rm -rf models

.PHONY: \
	update \
	update.vendor \
	build \
	build.cuda \
	build.openblas \
	test \
	docker \
	run-server \
	run-example-docker \
	clean
