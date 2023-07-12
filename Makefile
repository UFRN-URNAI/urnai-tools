# Variables
PACKAGE_NAME=urnai
ROOT=$(shell pwd)

## Lint
DOCKER_LINTER_IMAGE=urnai/linter:latest
LINT_COMMIT_TARGET_BRANCH=origin/v2

## Test
TEST_CONTAINER_NAME=${PACKAGE_NAME}_test
TEST_COMMAND=coverage run -m pytest tests

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
	@docker run --rm -v ${ROOT}:/app ${DOCKER_LINTER_IMAGE} " \
		lint-commit ${LINT_COMMIT_TARGET_BRANCH} \
		&& lint-markdown \
		&& lint-yaml \
		&& lint-shell-script \
		&& lint-dockerfile \
		&& lint-python urnai"

.PHONY: test
test:
	@docker run --rm -v ${ROOT}:/app \
		--name ${TEST_CONTAINER_NAME} ${PACKAGE_NAME} \
		${TEST_COMMAND}

.PHONY: test-coverage
test-coverage:
	@docker run --rm -v ${ROOT}:/app \
		--name ${TEST_CONTAINER_NAME} ${PACKAGE_NAME} \
		/bin/bash -c "${TEST_COMMAND} && coverage report -m"

.PHONY: shell
shell:
	@docker run --rm -it -v ${ROOT}:/app ${PACKAGE_NAME} /bin/bash
