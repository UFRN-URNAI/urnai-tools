# Development of the URNAI package

For the development of URNAI, we used the following tools:

- [Docker][docker]
- [Micromamba][micromamba]
- [Taskfile][taskfile]

You can use the Micromamba to manage your virtual environment to run
the scripts or use the Docker image for the same purpose.

## Virtual environment

:warning: To use the virtual environment, you need to have Micromamba
installed on your machine.

Create the virtual environment:

```shell
micromamba create -n urnai_env -f environment.yml -y
```

Activate the virtual environment:

```shell
micromamba activate urnai_env
```

Now you can develop URNAI on your machine.

## Docker

:warning: To use the Docker image, you must have Docker and Taskfile
installed on your machine.

Create the URNAI image:

```shell
task build
# Or
task build-no-cache
```

Created the image, you can enter a container and run the scripts
inside the container:

```shell
# Entering the container
task shell
# Running script
python path/to/your/file.py
```

A faster way to run a script is to use the `run` command:

```shell
task run -- path/to/your/file.py
```

Now you can develop URNAI on your machine.

[docker]: https://www.docker.com/
[micromamba]: https://mamba.readthedocs.io/en/latest/user_guide/micromamba.html
[taskfile]: https://taskfile.dev/
