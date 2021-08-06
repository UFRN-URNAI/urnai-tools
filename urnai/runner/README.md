# Command Line Interface (CLI)

The command line interface is available when URNAI is installed using pip. There are a set of commands with optional parameters that can be used to help the user with agent training. Basically, all these commands are distributed in labeled sets. For example, all training related commands are available under the 'train' label. So, the main concept behind URNAI CLI is to always call ``urnai <set-name>``. 

For a full list of command sets and their parameters, use: ``urnai -h``.

## Available Commands

### Train Set

- General Description

The basic command set present in URNAI Tools is ``urnai train``. This is the command used to start a training and convert training files.

- Command examples:

To start a new training with a JSON or CSV solve file:

```
urnai train --train-file <TRAIN_FILE_PATH>
```

To start a new play session (training session will be ignored by URNAI):

```
urnai train --play --train-file <TRAIN_FILE_PATH>
```

To convert a JSON file to CSV:

```
urnai train --convert <TRAIN_FILE_PATH> --output-format CSV
```

To convert a CSV file to JSON:

```
urnai train --convert <TRAIN_FILE_PATH> --output-format JSON
```

### SC2 Set

- General Description

The SC2 set is aimed to help developers extract useful information from the StarCraft II environment. Currently, there is a function that is capable of extracting all feature maps from pysc2. 

- Command example:

To extract all feature layers from <SC2_MAP> to current directory (output will be in CSV format):

```
urnai sc2 --sc2-map <SC2_MAP>
```

### DRTS Set

- General Description

Similarly, DRTS set is composed by functions which are capable of interacting with the DeepRTS environment. There are several triggers aimed to help dealing with maps. 

- Command examples:

To view a specific DeepRTS map, write the command below. It will open the game and freeze it at step 0. This way, the map can be viewed rendered by the game. Ctrl + C (on the terminal) to cancel the view.


```
urnai drts --drts-map <DRTS_MAP_NAME>.json
```

To extract the map and its features to the current folder:

```
urnai drts --extract-specs --drts-map <DRTS_MAP_NAME>.json
```

To build (or re-build) the map using the features inside the specified folder:

```
urnai drts --build-map --drts-map-specs <DRTS_MAP_SPECS_DIR>
```

To install a map into the game default directory:

```
urnai drts --install --drts-map <MAP_FILE_PATH>
```

To uninstall a map from the game directory:

```
urnai drts --uninstall --drts-map <MAP_NAME>.json
```

To see which maps are installed:

```
urnai drts --show-available-maps
```

## Adding new commands

For someone familiarized with the Python language, adding new commands should be relatively simple. The developer should pay attention to two files: [commands.py](https://github.com/UFRN-URNAI/urnai-tools/blob/master/urnai/runner/commands.py) and [runnerbuilder.py](https://github.com/UFRN-URNAI/urnai-tools/blob/master/urnai/runner/runnerbuilder.py). One set of commands should be a new class inside commands.py file. After coding the class, its name should be registered in runnerbuilder.py.

It is advised to the reader to take a look at both mentioned files' code while reading this howto, as it is heavily based on those.

### How to do it

As it was mentioned, by taking a look into the commands.py file and using one of the classes inside, the reader should have a nice basis. In general, the class' name could follow the standard of the file. For example, drts commands are present in DeepRTSRunner class, sc2 commands are present in SC2Runner class, and so on.

After choosing a name, the programmer should fill the two *mandatory* constants: COMMAND and OPT_COMMANDS. COMMAND carries the name of the set. Whilst OPT_COMMANDS are the triggers that represent the functions _per se_. Just take a look at DeepRTSRunner and it should give one an idea on how to write these two variables.

Then there is the method run(). This is what is called after the object of the command set class is created. Here all the code of identifying triggers and function calling is written to make the command work. As this guide is heavily based on the code of commands.py, the reader should take a look at the code to get an idea on how to manipulate parameters and call functions to execute the command routine.

After the class is done, its name *must* be registered inside the [runnerbuilder.py](https://github.com/UFRN-URNAI/urnai-tools/blob/master/urnai/runner/runnerbuilder.py) file. There, there is the constant COMMANDS, where the name of your class should be.

Finally, to test your code, URNAI must be reinstalled inside your virtualenv and tested by calling it on the command line.
