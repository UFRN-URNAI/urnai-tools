# Super-computing

URNAI can run in super-computing environments. For now, we support running the library by using Singulrity containers. On this folder you will find a singularity recipe (simg file), a script to build the container (sif file) and a config.json.example to configure the script parameters. 

## Dependencies

You need to have Singularity installed. Go [here](https://sylabs.io/guides/3.8/user-guide/quick_start.html#quick-installation-steps) for instructions.

You also need to be running a NVidia gpu. For now, the recipe present on this directory does not use tensorflow-cpu inside the image. Nonetheless, it is on our roadmap to support it.

## Building the image

The image is built with all the needed basic libraries to run trainings on the StarCraft II environment. To start the building process, take a look at the config.json.example file, change it to suit your needs (do *NOT* use relative paths, always use full file paths) and rename it to config.json. After this, run on a terminal:

```
$ chmod +x build_sif.sh
$ ./build_sif.sh
```

## Start training using the built image

Run this on the terminal:

```
    SINGULARITYENV_PYTHONIOENCODING=utf8 singularity exec --nv IMAGE_NAME.sif urnai train --train-file TRAIN_FILE_NAME.json 

```

## Tips

- If you run your trainings in a super-computing cluster, use the command above inside your slurm script to run the training.

- Singularity automatically mounts your home folder inside the container. So if you need any files for your training (i.e. StarCraft II installation). Copy them to your home folder and access them using their full paths in the singularity command.
