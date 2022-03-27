# Variables
DOCKER_LINTER_IMAGE=urnai/linter:latest
ROOT=$(shell pwd)
LINT_COMMIT_TARGET_BRANCH=origin/master

# Commands
.PHONY: install-hooks
install-hooks:
	git config core.hooksPath .githooks

.PHONY: lint
lint:
	@docker pull ${DOCKER_LINTER_IMAGE}
	@docker run --rm -v ${ROOT}:/app ${DOCKER_LINTER_IMAGE} " \
		lint-commit ${LINT_COMMIT_TARGET_BRANCH} \
		&& lint-markdown \
		&& lint-python"
