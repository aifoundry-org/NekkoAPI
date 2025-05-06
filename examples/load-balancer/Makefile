VERSION ?= $(shell git describe --tags --always --dirty 2>/dev/null || echo "dev")
COMMIT ?= $(shell git rev-parse --short HEAD 2>/dev/null || echo "unknown")
BUILD_DATE ?= $(shell date -u +"%Y-%m-%dT%H:%M:%SZ")

APP_NAME := aifoundryorg/load-balancer
IMAGE_NAME := $(APP_NAME)
DOCKER_TAG := $(IMAGE_NAME):$(VERSION)

.PHONY: deps
deps:
	go mod tidy

.PHONY: build
build: deps
	docker build -t $(DOCKER_TAG) -f Dockerfile .

.PHONY: run
run:
	docker run --rm $(DOCKER_TAG)

.PHONY: clean
clean:
	go clean --modcache
	docker rmi $(DOCKER_TAG) || true
