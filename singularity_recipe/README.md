# Super-computing

URNAI can run in super-computing environments. For now, we support running the library by using Singulrity containers. On this folder you will find a singularity recipe (simg file), a script (sh file) to build the container (sif file) and a config.json.example to configure the script parameters. 

## Summary

- [Dependencies](#dependencies)
- [Building the image](#building)
- [Start training using the built image](#start)
- [Installing Singularity on Windows Subsystem for Linux](#installing)
- [Editing SIF Files (Editing Images)](#editing)
- [Tips](#tips)

## <a name="dependencies"></a>Dependencies

You need to have Singularity installed. Go [here](https://sylabs.io/guides/3.8/user-guide/quick_start.html#quick-installation-steps) for instructions.

You also need to be running a NVidia gpu. For now, the recipe present on this directory does not use tensorflow-cpu inside the image. Nonetheless, it is on our roadmap to support it.

## <a name="building"></a>Building the image

The image is built with all the needed basic libraries to run trainings on the StarCraft II environment. To start the building process, take a look at the config.json.example file, change it to suit your needs (do *NOT* use relative paths, always use full file paths) and rename it to config.json. After this, run on a terminal:

```
$ chmod +x build_sif.sh
$ ./build_sif.sh
```

## <a name="start"></a>Start training using the built image

Run this on the terminal:

```
    SINGULARITYENV_PYTHONIOENCODING=utf8 singularity exec --nv IMAGE_NAME.sif urnai train --train-file TRAIN_FILE_NAME.json 

```
## <a name="installing"></a>Installing Singularity on Windows Subsystem for Linux

Not everyone uses Linux for developing AI with Python. There are some users who prefer to use Windows instead. Sometimes, these users need to build the Singularity Image (sif file) to setup the needed environment for their trainings.

In these cases, Singularity can be used to make these images on a Windows environment. Here we have a quick guide on how to install it on Windows. First of all the reader should know that these instructions are about the same as the official [Singularity installation instrucions](https://sylabs.io/guides/3.8/user-guide/quick_start.html#quick-installation-steps).

The first thing to do is to install WSL (Windows Subsystem for Linux), a container technology that does not only allow to run Linux inside Windows seamlessly, but also to run it with almost native speeds. To proceed to the installation, the reader should refer to [the official installation steps on Microsoft documentation](https://docs.microsoft.com/en-us/windows/wsl/install).

After everything is set up, get a terminal under your chosen distribution (default is Ubuntu) and go through the official installation guide or follow this one up. It's up to you. 

First off, update all Ubuntu packages:

```
sudo apt update && sudo apt upgrade -y
```

Now, to the installation steps. Export some environment variables to help with the next commands (you can also adjust the values to your context):

```
export SINGULARITY_VERSION=3.8.1
export GO_VERSION=1.17.2 
export OS=linux 
export ARCH=amd64
```

Install singularity dependencies:

```
sudo apt install -y \
    build-essential \
    libssl-dev \
    uuid-dev \
    libgpgme11-dev \
    squashfs-tools \
    libseccomp-dev \
    wget \
    pkg-config \
    git \
    cryptsetup
```

After those dependencies, Go is needed, so Singularity can be compiled for the WSL container:

```
wget https://dl.google.com/go/go$GO_VERSION.$OS-$ARCH.tar.gz && \
  sudo tar -C /usr/local -xzvf go$GO_VERSION.$OS-$ARCH.tar.gz && \
  rm go$GO_VERSION.$OS-$ARCH.tar.gz    
```

Include Go on PATH and persist this change:

```
export PATH=/usr/local/go/bin:$PATH
echo 'export PATH=/usr/local/go/bin:$PATH' >> ~/.bashrc
```

Now download and install Singularity. This process may take a while, since Singularity will be compiled in the process:

```
wget https://github.com/sylabs/singularity/releases/download/v$SINGULARITY_VERSION/singularity-ce-$SINGULARITY_VERSION.tar.gz && \
tar -xzf singularity-ce-$SINGULARITY_VERSION.tar.gz && \
cd singularity-ce-$SINGULARITY_VERSION
./mconfig && \
make -C builddir && \
sudo make -C builddir install
```

You can now use Singularity on your Linux environment to build images using this guide!

## <a name="editing"></a>Editing SIF Files (Editing Images)

Sometimes it is needed to edit an image file. This happens when the user wants to update something inside it without going through the process of rebuilding the image again, which sometimes can take several minutes. 

In our context, for example, it is very often that we need to update URNAI to the latest version in the sif file. To do that, one just need to get a shell inside the container, with write permissions. This means that every file you create there will be saved inside the sif file, including anything you install using pip or apt, for example. To do this, go to the folder where your container image is, and ask singularity to get a shell with write permissions:

```
sudo singularity shell --writable {YOUR_SIF_FILE_NAME}.sif
```

## <a name="tips"></a>Tips

- If you run your trainings in a super-computing cluster, use the command above inside your slurm script to run the training.

- Singularity automatically mounts your home folder inside the container. So if you need any files for your training (i.e. StarCraft II installation). Copy them to your home folder and access them using their full paths in the singularity command.
