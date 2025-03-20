APP_NAME := nekko-lb
IMAGE_NAME := $(APP_NAME)
VERSION := 0.1.0
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
