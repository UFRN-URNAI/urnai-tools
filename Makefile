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
	@cp requirements.txt ./docker/test
	@docker build -t ${PACKAGE_NAME} ./docker/test
	@rm ./docker/test/requirements.txt

.PHONY: build-no-cache
build-no-cache: install-hooks
	@cp requirements.txt ./docker/test
	@docker build --no-cache -t ${PACKAGE_NAME} ./docker/test
	@rm ./docker/test/requirements.txt

.PHONY: build-full
build-full: install-hooks
	@cp requirements.txt ./docker/full
	@docker build -t ${PACKAGE_NAME} ./docker/full
	@rm ./docker/full/requirements.txt


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
		/bin/bash -c "pip3 install --no-dependencies /app && ${TEST_COMMAND}"

.PHONY: test-coverage
test-coverage:
	@docker run --rm -v ${ROOT}:/app \
		--name ${TEST_CONTAINER_NAME} ${PACKAGE_NAME} \
		/bin/bash -c "pip3 install --no-dependencies /app && ${TEST_COMMAND} && coverage report -m"

.PHONY: shell
shell:
	@docker run --rm -it -v ${ROOT}:/app ${PACKAGE_NAME} /bin/bash
