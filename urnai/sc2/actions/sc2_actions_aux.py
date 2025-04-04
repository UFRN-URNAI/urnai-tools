from pysc2.env import sc2_env
from pysc2.lib import actions, features, units

"""
An action set defines all actions an agent can use. In the case of StarCraft 2 
using PySC2, some actions require extra processing to work, so it's up to the 
developper to come up with a way to make them work.

Even though this is not called an action_wrapper, it effectively acts as a wrapper

e.g: actions.RAW_FUNCTIONS.Build_Barracks_pt is a function implemented in PySC2 that
requires some extra arguments to work, like whether to build it now or to queue the 
action, which worker is going to perform this action, and the target (given by a
[x, y] position)

In this file we sort all of this issues, like deciding when to do an action, which 
units to use for it, and where to build.
The methods in here effectively serve as a bridge between our high level actions
defined in sc2_wrapper.py and the PySC2 library.
"""

"""CONSTANTS USED TO DO GENERAL CHECKS"""
_TERRAN = sc2_env.Race.terran
_PROTOSS = sc2_env.Race.protoss
_ZERG = sc2_env.Race.zerg


def no_op():
    return actions.RAW_FUNCTIONS.no_op()


"""
The following methods are used to aid in various mechanical operations the agent has 
to perform, such as: getting all units from a certain type, counting the amount of 
free supply, etc
"""


def get_units_by_type(obs, unit_type, alliance=features.PlayerRelative.SELF):
    return [unit for unit in obs.raw_units
            if unit.unit_type == unit_type
            and unit.alliance == alliance
            and unit.build_progress == 100]


def get_neutral_units_by_type(obs, unit_type):
    return [unit for unit in obs.raw_units
            if unit.unit_type == unit_type
            and unit.alliance == features.PlayerRelative.NEUTRAL]


def get_all_neutral_units(obs):
    return [unit for unit in obs.raw_units
            if unit.alliance == features.PlayerRelative.NEUTRAL]


def get_army_unit_types(player_race):

    army_unit_types = []

    if player_race == _PROTOSS:
        army_unit_types = [
            units.Protoss.Adept, units.Protoss.AdeptPhaseShift, units.Protoss.Archon,
            units.Protoss.Carrier, units.Protoss.Colossus, units.Protoss.DarkTemplar,
            units.Protoss.Disruptor, units.Protoss.DisruptorPhased, 
            units.Protoss.HighTemplar, units.Protoss.Immortal, units.Protoss.Mothership,
            units.Protoss.Observer, units.Protoss.ObserverSurveillanceMode, 
            units.Protoss.Oracle, units.Protoss.Phoenix, units.Protoss.Sentry, 
            units.Protoss.Stalker, units.Protoss.Tempest,
            units.Protoss.VoidRay, units.Protoss.Zealot,
        ]
    elif player_race == _TERRAN:
        army_unit_types = [units.Terran.Marine, units.Terran.Marauder, 
                           units.Terran.Reaper, units.Terran.Ghost, 
                           units.Terran.Hellion, units.Terran.Hellbat, 
                           units.Terran.SiegeTank, units.Terran.Cyclone, 
                           units.Terran.WidowMine, units.Terran.Thor, 
                           units.Terran.ThorHighImpactMode, units.Terran.VikingAssault, 
                           units.Terran.VikingFighter, units.Terran.Medivac, 
                           units.Terran.Liberator, units.Terran.LiberatorAG, 
                           units.Terran.Raven, units.Terran.Banshee, 
                           units.Terran.Battlecruiser]
    elif player_race == _ZERG:
        army_unit_types = [
            units.Zerg.Baneling, units.Zerg.BanelingBurrowed, units.Zerg.BanelingCocoon,
            units.Zerg.BroodLord, units.Zerg.BroodLordCocoon, units.Zerg.Broodling, 
            units.Zerg.BroodlingEscort, units.Zerg.Changeling, 
            units.Zerg.ChangelingMarine, units.Zerg.ChangelingMarineShield,
            units.Zerg.ChangelingZealot, units.Zerg.ChangelingZergling,
            units.Zerg.ChangelingZerglingWings, units.Zerg.Corruptor, 
            units.Zerg.Hydralisk, units.Zerg.HydraliskBurrowed, units.Zerg.Infestor,
            units.Zerg.InfestorBurrowed, units.Zerg.Locust, units.Zerg.LocustFlying, 
            units.Zerg.Lurker, units.Zerg.LurkerBurrowed, units.Zerg.LurkerCocoon, 
            units.Zerg.Mutalisk, units.Zerg.Overseer, units.Zerg.OverseerCocoon,
            units.Zerg.OverseerOversightMode, units.Zerg.Queen, 
            units.Zerg.QueenBurrowed, units.Zerg.Ravager, units.Zerg.RavagerBurrowed,
            units.Zerg.RavagerCocoon, units.Zerg.Roach, units.Zerg.RoachBurrowed,
            units.Zerg.SwarmHost, units.Zerg.SwarmHostBurrowed,
            units.Zerg.Ultralisk, units.Zerg.UltraliskBurrowed, units.Zerg.Viper,
            units.Zerg.Zergling, units.Zerg.ZerglingBurrowed,
        ]
    
    return army_unit_types

def select_army(obs, player_race):
    army = []
    army_unit_types = get_army_unit_types(player_race)

    army = [unit for unit in obs.raw_units if
            unit.alliance == features.PlayerRelative.SELF \
                and unit.unit_type in army_unit_types]

    return army

def select_enemy_army(obs, race = None):
    army = []

    if race is None:
        army = [unit for unit in obs.raw_units if
                unit.alliance == features.PlayerRelative.ENEMY]
    else:
        army_unit_types = get_army_unit_types(race)
        army = [unit for unit in obs.raw_units if
            unit.alliance == features.PlayerRelative.SELF \
                and unit.unit_type in army_unit_types]
    
    return army