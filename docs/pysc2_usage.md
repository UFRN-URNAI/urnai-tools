# PySC2 usage in URNAI

## Actions in URNAI vs Actions in PySC2

In PySC2 you would normally call an action in the following manner:
`actions.X.F`. Where `X` is the action function set, such as `RAW_FUNCTIONS`
or `FUNCTIONS`, and `F` is the action function you are calling,
such as `no_op` or `Move_pt`.

But in URNAI, due to encapsulation, the call is done in the following manner:
`X[F].run()`. Where `X` is a dict containing classes for each of the action functions
in PySC2, and `F` is a string with the name of the action function you are calling.
The dictionaries mentioned above are stored in the `urnai/sc2/actions/sc2_actions.py`, and can
be imported such as in the example below:

```python
from urnai.sc2 import raw_functions_classes as sc2_actions
```

### Examples

```python
# PySC2
actions.RAW_FUNCTIONS.no_op()
actions.RAW_FUNCTIONS.Move_pt('now', unit.tag, [new_army_x, new_army_y])

# URNAI
sc2_actions["no_op"].run()
sc2_actions["Move_pt"].run('now', unit.tag,[new_army_x, new_army_y])
```

### Why?

URNAI chooses to represent actions as classes so they can be better organized and tested.
Therefore, this encapsulation, which transforms each PySC2 action into a class, contributes
to all environments being done under a single system, something that is desired.
