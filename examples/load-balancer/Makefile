APP_NAME := nekko-lb
IMAGE_NAME := $(APP_NAME)
PLATFORM := linux/amd64

.PHONY: deps
deps:
	GO111MODULE=on go install github.com/google/ko@latest
	go mod tidy

.PHONY: build
build: deps
	KO_DOCKER_REPO=$(IMAGE_NAME) $(HOME)/go/bin/ko build --platform=$(PLATFORM) --bare --local

.PHONY: clean
clean:
	go clean --modcache

