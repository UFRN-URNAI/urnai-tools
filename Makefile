#Variables
PACKAGE_NAME=urnai
ROOT=$(shell pwd)

## Lint
DOCKER_LINTER_IMAGE=urnai/linter:latest
LINT_COMMIT_TARGET_BRANCH=origin/master

## Test
TEST_CONTAINER_NAME=${PACKAGE_NAME}_test
TEST_COMMAND=coverage run -m pytest tests

# Commands
.PHONY: build
build: install-hooks
	@cp requirements.txt ./dockerfiles/test
	@docker build -t ${PACKAGE_NAME} ./dockerfiles/test
	@rm ./dockerfiles/test/requirements.txt

.PHONY: build-no-cache
build-no-cache: install-hooks
	@cp requirements.txt ./dockerfiles/test
	@docker build --no-cache -t ${PACKAGE_NAME} ./dockerfiles/test
	@rm ./dockerfiles/test/requirements.txt

.PHONY: build-full
build-full: install-hooks
	@cp requirements.txt ./dockerfiles/full
	@docker build --no-cache -t ${PACKAGE_NAME} ./dockerfiles/full
	@rm ./dockerfiles/full/requirements.txt


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
