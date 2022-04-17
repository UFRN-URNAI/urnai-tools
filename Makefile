# Variables
PACKAGE_NAME=urnai
ROOT=$(shell pwd)

## Lint
DOCKER_LINTER_IMAGE=urnai/linter:latest
LINT_COMMIT_TARGET_BRANCH=origin/master

# Commands
.PHONY: build
build: install-hooks
	@docker build -t ${PACKAGE_NAME} .

.PHONY: build-no-cache
build-no-cache: install-hooks
	@docker build --no-cache -t ${PACKAGE_NAME} .

.PHONY: install-hooks
install-hooks:
	git config core.hooksPath .githooks

.PHONY: lint
lint:
	@docker pull ${DOCKER_LINTER_IMAGE}
	@docker run --rm -v ${ROOT}:/app ${DOCKER_LINTER_IMAGE} " \
		lint-commit ${LINT_COMMIT_TARGET_BRANCH} \
		&& lint-markdown \
		&& lint-python urnai"

.PHONY: shell
shell:
	@docker run --rm -it -v ${ROOT}:/app ${PACKAGE_NAME} /bin/bash
