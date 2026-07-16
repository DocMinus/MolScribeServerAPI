# Build / publish helpers for the MolScribe server (CPU) image.
# The CPU image uses the PyTorch CPU-only wheels, which are published for x86_64, so it is a
# linux/amd64 image. (The GPU image is built via docker-compose.cuda.yml / Dockerfile.cuda.)
# edit .env if you want to push to your own dockerhub account
-include .env

IMAGE_NAME = molscribe_api
DOCKER_USER ?= docminus
REGISTRY ?= docker.io
IMAGE_REF := $(REGISTRY)/$(DOCKER_USER)/$(IMAGE_NAME)
PLATFORM ?= linux/amd64
VERSION ?=

.PHONY: all build buildx tag push publish check-version freeze-latest release

all: buildx

publish: buildx tag push

check-version:
	@if [ -z "$(VERSION)" ]; then \
		echo "ERROR: VERSION is required (example: make release VERSION=0.1.0)"; \
		exit 1; \
	fi

build:
	@echo "\n------------------------------------"
	@echo "Building the image $(IMAGE_NAME):"
	docker build -t $(IMAGE_NAME) .
	@echo "------------------------------------\n"
	@echo "Use 'docker compose up -d' to start the container"

# Local build loaded into the local docker (linux/amd64; runs on Apple Silicon via emulation).
buildx:
	@echo "\n------------------------------------"
	@echo "Building the image $(IMAGE_NAME) ($(PLATFORM), loaded locally):"
	docker buildx build --platform $(PLATFORM) --load -t $(IMAGE_NAME) .
	@echo "------------------------------------\n"
	@echo "Use 'docker compose up -d' to start the container"

tag:
	@echo "\n------------------------------------"
	@echo "Tagging the image $(IMAGE_NAME):"
	docker tag $(IMAGE_NAME) $(IMAGE_REF):latest
	@echo "------------------------------------\n"

push:
	@echo "\n------------------------------------"
	@echo "Pushing the image $(IMAGE_NAME):"
	docker push $(IMAGE_REF):latest
	@echo "------------------------------------\n"

freeze-latest: check-version
	@echo "\n------------------------------------"
	@echo "Freezing current latest as $(VERSION):"
	docker buildx imagetools create --tag $(IMAGE_REF):$(VERSION) $(IMAGE_REF):latest
	@echo "------------------------------------\n"

# Multi-arch build + push of both a version tag and latest.
release: check-version
	@echo "\n------------------------------------"
	@echo "Building and pushing $(VERSION) and latest ($(PLATFORM)):"
	docker buildx build --platform $(PLATFORM) -t $(IMAGE_REF):$(VERSION) -t $(IMAGE_REF):latest --push .
	@echo "------------------------------------\n"
