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

To start a new play session (training senssion will be ignored by URNAI):

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

The SC2 set is a group of tools aimed to help developers extract useful information from the StarCraft II environment. Currently, there is only a function that is capable of extracting all feature maps from pysc2. As we detect more use cases we will be adding more functions.

- Command examples:

To extract all feature layers from <SC2_MAP> to current directory (output will be in CSV format):

```
urnai sc2 --sc2-map <SC2_MAP>
```

