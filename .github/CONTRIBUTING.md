# Contributing to URNAI

All contributions, bug reports, bug fixes, documentation improvements, enhancements,
and ideas are welcome.

## Development environment

We use Docker as a containerization tool, so you will need to build the project:

```shell
task build-no-cache
```

If you want to generate an image with your modifications, but without installing
dependencies, you can run the command below:

```shell
task build
```

## Accessing the container

After creating the project image, you can go inside it to run specific scripts:

```shell
task shell
```

### Linter

To check the syntax of the written code, documentation, configuration files,
commits, etc, run the command below:

```shell
task lint
```

### Unit tests

To run the unit tests locally, simply run the command below:

```shell
task test
```

## Step by step for contribution

1. Open a Pull Request (PR) with your modifications.
   1. Every dependency addition must be described in the PR.
   2. If necessary, the PR must contain the tests and documentations of the modifications.
2. Your PR must go through the CI jobs.
3. A URNAI maintainer must review your PR and approve it.
4. Once approved, your PR will enter the `main` branch and be released in the next update.

### Branch naming

We recommend that the branch is linked to an issue.
This way the branch nomenclature should be: `issue-<issue_id>`.
(example: for issue 10, the branch should be `issue-10`).

### Commit

We follow the specifications of Conventional Commits.
For more details, just go to [official documentation][conventional-commits].

### Code review

Your PR will be reviewed by an URNAI maintainer.
It is interesting that your PR contains all the information needed for the review,
in addition to unit tests and documentation. Just imagine that someone needs to know everything
what you did, how you did it and how to test it without having to ask you directly.

[conventional-commits]: https://www.conventionalcommits.org/en/v1.0.0/
