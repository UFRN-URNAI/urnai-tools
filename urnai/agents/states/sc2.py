import math
import numpy as np
import pysc2
from s2clientprotocol.common_pb2 import Terran
import scipy.misc
from matplotlib import colors
from matplotlib import pyplot as plt
from .abstate import StateBuilder
from pysc2.lib import actions, features, units
from urnai.agents.actions.sc2 import *
from pysc2.env import sc2_env
from urnai.utils.image import *
from urnai.utils.sc2_utils import get_fog_of_war_percentage


class Simple64State(StateBuilder):

    def __init__(self, reduction_factor=4):
        self.reduction_factor = reduction_factor

        self._state_size = int(22 + (64/self.reduction_factor)**2)
        self.player_race = 0

    def build_state(self, obs):
        if obs.game_loop[0] < 80 and self.base_top_left == None:

            commandcenter = get_units_by_type(obs, units.Terran.CommandCenter)
            nexus = get_units_by_type(obs, units.Protoss.Nexus)
            hatchery = get_units_by_type(obs, units.Zerg.Hatchery)
            if len(commandcenter)>0: 
                townhall = commandcenter[0]
                self.player_race = sc2_env.Race.terran
            if len(nexus)>0:
                townhall = nexus[0]
                self.player_race = sc2_env.Race.protoss
            if len(hatchery)>0:
                townhall = hatchery[0]
                self.player_race = sc2_env.Race.zerg

            self.base_top_left = (townhall.x < 32)

        new_state = []
        new_state.append(obs.player.minerals/6000)
        new_state.append(obs.player.vespene/4000)
        new_state.append(obs.player.food_cap/200)
        new_state.append(obs.player.food_used/200)
        new_state.append(obs.player.food_army/200)
        new_state.append(obs.player.food_workers/200)
        new_state.append((obs.player.food_cap - obs.player.food_used)/200)
        new_state.append(obs.player.army_count/200)
        new_state.append(obs.player.idle_worker_count/200)

        if self.player_race == sc2_env.Race.terran:
            new_state.append(get_my_units_amount(obs, units.Terran.CommandCenter)+
                            get_my_units_amount(obs, units.Terran.OrbitalCommand)+
                            get_my_units_amount(obs, units.Terran.PlanetaryFortress)/2)
            new_state.append(get_my_units_amount(obs, units.Terran.SupplyDepot)/8)
            new_state.append(get_my_units_amount(obs, units.Terran.Refinery)/4)
            new_state.append(get_my_units_amount(obs, units.Terran.EngineeringBay))
            new_state.append(get_my_units_amount(obs, units.Terran.Armory))
            new_state.append(get_my_units_amount(obs, units.Terran.MissileTurret)/8)
            new_state.append(get_my_units_amount(obs, units.Terran.SensorTower)/3)
            new_state.append(get_my_units_amount(obs, units.Terran.Bunker)/5)
            new_state.append(get_my_units_amount(obs, units.Terran.FusionCore))
            new_state.append(get_my_units_amount(obs, units.Terran.GhostAcademy))
            new_state.append(get_my_units_amount(obs, units.Terran.Barracks)/3)
            new_state.append(get_my_units_amount(obs, units.Terran.Factory)/2)
            new_state.append(get_my_units_amount(obs, units.Terran.Starport)/2)

        elif self.player_race == sc2_env.Race.protoss:
            new_state.append(get_my_units_amount(obs, units.Protoss.Nexus))
            new_state.append(get_my_units_amount(obs, units.Protoss.Pylon))
            new_state.append(get_my_units_amount(obs, units.Protoss.Assimilator))
            new_state.append(get_my_units_amount(obs, units.Protoss.Forge))
            new_state.append(get_my_units_amount(obs, units.Protoss.Gateway))
            new_state.append(get_my_units_amount(obs, units.Protoss.CyberneticsCore))
            new_state.append(get_my_units_amount(obs, units.Protoss.PhotonCannon))
            new_state.append(get_my_units_amount(obs, units.Protoss.RoboticsFacility))
            new_state.append(get_my_units_amount(obs, units.Protoss.Stargate))
            new_state.append(get_my_units_amount(obs, units.Protoss.TwilightCouncil))
            new_state.append(get_my_units_amount(obs, units.Protoss.RoboticsBay))
            new_state.append(get_my_units_amount(obs, units.Protoss.TemplarArchive))
            new_state.append(get_my_units_amount(obs, units.Protoss.DarkShrine))
            
        elif self.player_race == sc2_env.Race.zerg:
            new_state.append(get_my_units_amount(obs, units.Zerg.BanelingNest))
            new_state.append(get_my_units_amount(obs, units.Zerg.EvolutionChamber))
            new_state.append(get_my_units_amount(obs, units.Zerg.Extractor))
            new_state.append(get_my_units_amount(obs, units.Zerg.Hatchery))
            new_state.append(get_my_units_amount(obs, units.Zerg.HydraliskDen))
            new_state.append(get_my_units_amount(obs, units.Zerg.InfestationPit))
            new_state.append(get_my_units_amount(obs, units.Zerg.LurkerDen))
            new_state.append(get_my_units_amount(obs, units.Zerg.NydusNetwork))
            new_state.append(get_my_units_amount(obs, units.Zerg.RoachWarren))
            new_state.append(get_my_units_amount(obs, units.Zerg.SpawningPool))
            new_state.append(get_my_units_amount(obs, units.Zerg.SpineCrawler))
            new_state.append(get_my_units_amount(obs, units.Zerg.Spire))
            new_state.append(get_my_units_amount(obs, units.Zerg.SporeCrawler))


        m0 = obs.feature_minimap[0]     # Feature layer of the map's terrain (elevation and shape)
        m0 = m0/10

        m1 = obs.feature_minimap[2]     # Feature layer of creep in the minimap (generally will be quite empty, especially on games without zergs hehe)
        m1[m1 == 1] = 16                # Transforming creep info from 1 to 16 in the map (makes it more visible)

        m2 = obs.feature_minimap[4]     # Feature layer of all visible units (neutral, friendly and enemy) on the minimap
        m2[m2 == 1] = 128               # Transforming own units from 1 to 128 for visibility
        m2[m2 == 3] = 64                # Transforming enemy units from 3 to 64 for visibility
        m2[m2 == 2] = 64
        m2[m2 == 16] = 32               # Transforming mineral fields and geysers from 16 to 32 for visibility               
        
        combined_minimap = np.where(m2 != 0, m2, m1)                                    # Overlaying the m2(units) map over the m1(creep) map
        combined_minimap = np.where(combined_minimap != 0, combined_minimap, m0)        # Overlaying the combined m1 and m2 map over the m0(terrain) map
        # combined_minimap = m0+m1+m2
        combined_minimap = np.array(combined_minimap)                                   # Tranforming combined_minimap into a np array so tensor flow can interpret it
        combined_minimap = combined_minimap/combined_minimap.max()                      # Normalizing between 0 and 1

        # Lowering the featuremap's resolution
        lowered_minimap = lower_featuremap_resolution(combined_minimap, self.reduction_factor)      #featuremap and reduction factor, if rf = 4 a 64x64 map will be transformed into a 16x16 map
        
        # Rotating observation depending on Agent's location on the map so that we get a consistent, generalized, observation
        if not self.base_top_left: 
            lowered_minimap = np.rot90(lowered_minimap, 2)
        
        new_state.extend(lowered_minimap.flatten())   
        final_state = np.array(new_state)
        final_state = np.expand_dims(final_state, axis=0)


        # Displaying the agent's vision in a heatmap using matplotlib every 200 steps (just for debug purpuses, probably will be removed later)
        # if (obs.game_loop[0]/16)%300 == 0:
        #     # Displaying Agent's vision
        #     #norm = colors.Normalize()
        #     plt.figure()
        #     #plt.imshow(np.array(obs.feature_minimap[0]))
        #     #plt.imshow(np.array(obs.feature_minimap[2]))
        #     #plt.imshow(np.array(obs.feature_minimap[4]))
        #     #plt.imshow(np.array(combined_minimap))
        #     plt.imshow(np.array(mtest))
        #     plt.show()

        return final_state


    def get_state_dim(self):
        return self._state_size

    
class Simple64StateFullRes(StateBuilder):

    def __init__(self, reduction_factor=2):
        #self._state_size = 1957

        self.reduction_factor = reduction_factor
        self._state_size = int(22 + (44/self.reduction_factor)**2)
        self.player_race = 0

    def build_state(self, obs):
        if obs.game_loop[0] < 80 and self.base_top_left == None:

            commandcenter = get_units_by_type(obs, units.Terran.CommandCenter)
            nexus = get_units_by_type(obs, units.Protoss.Nexus)
            hatchery = get_units_by_type(obs, units.Zerg.Hatchery)
            if len(commandcenter)>0: 
                townhall = commandcenter[0]
                self.player_race = sc2_env.Race.terran
            if len(nexus)>0:
                townhall = nexus[0]
                self.player_race = sc2_env.Race.protoss
            if len(hatchery)>0:
                townhall = hatchery[0]
                self.player_race = sc2_env.Race.zerg

            self.base_top_left = (townhall.x < 32)

        new_state = []
        new_state.append(obs.player.minerals/6000)
        new_state.append(obs.player.vespene/4000)
        new_state.append(obs.player.food_cap/200)
        new_state.append(obs.player.food_used/200)
        new_state.append(obs.player.food_army/200)
        new_state.append(obs.player.food_workers/200)
        new_state.append((obs.player.food_cap - obs.player.food_used)/200)
        new_state.append(obs.player.army_count/200)
        new_state.append(obs.player.idle_worker_count/200)

        if self.player_race == sc2_env.Race.terran:
            new_state.append(get_my_units_amount(obs, units.Terran.CommandCenter)+
                            get_my_units_amount(obs, units.Terran.OrbitalCommand)+
                            get_my_units_amount(obs, units.Terran.PlanetaryFortress)/2)
            new_state.append(get_my_units_amount(obs, units.Terran.SupplyDepot)/8)
            new_state.append(get_my_units_amount(obs, units.Terran.Refinery)/4)
            new_state.append(get_my_units_amount(obs, units.Terran.EngineeringBay))
            new_state.append(get_my_units_amount(obs, units.Terran.Armory))
            new_state.append(get_my_units_amount(obs, units.Terran.MissileTurret)/8)
            new_state.append(get_my_units_amount(obs, units.Terran.SensorTower)/3)
            new_state.append(get_my_units_amount(obs, units.Terran.Bunker)/5)
            new_state.append(get_my_units_amount(obs, units.Terran.FusionCore))
            new_state.append(get_my_units_amount(obs, units.Terran.GhostAcademy))
            new_state.append(get_my_units_amount(obs, units.Terran.Barracks)/3)
            new_state.append(get_my_units_amount(obs, units.Terran.Factory)/2)
            new_state.append(get_my_units_amount(obs, units.Terran.Starport)/2)

        elif self.player_race == sc2_env.Race.protoss:
            new_state.append(get_my_units_amount(obs, units.Protoss.Nexus))
            new_state.append(get_my_units_amount(obs, units.Protoss.Pylon))
            new_state.append(get_my_units_amount(obs, units.Protoss.Assimilator))
            new_state.append(get_my_units_amount(obs, units.Protoss.Forge))
            new_state.append(get_my_units_amount(obs, units.Protoss.Gateway))
            new_state.append(get_my_units_amount(obs, units.Protoss.CyberneticsCore))
            new_state.append(get_my_units_amount(obs, units.Protoss.PhotonCannon))
            new_state.append(get_my_units_amount(obs, units.Protoss.RoboticsFacility))
            new_state.append(get_my_units_amount(obs, units.Protoss.Stargate))
            new_state.append(get_my_units_amount(obs, units.Protoss.TwilightCouncil))
            new_state.append(get_my_units_amount(obs, units.Protoss.RoboticsBay))
            new_state.append(get_my_units_amount(obs, units.Protoss.TemplarArchive))
            new_state.append(get_my_units_amount(obs, units.Protoss.DarkShrine))
            
        elif self.player_race == sc2_env.Race.zerg:
            new_state.append(get_my_units_amount(obs, units.Zerg.BanelingNest))
            new_state.append(get_my_units_amount(obs, units.Zerg.EvolutionChamber))
            new_state.append(get_my_units_amount(obs, units.Zerg.Extractor))
            new_state.append(get_my_units_amount(obs, units.Zerg.Hatchery))
            new_state.append(get_my_units_amount(obs, units.Zerg.HydraliskDen))
            new_state.append(get_my_units_amount(obs, units.Zerg.InfestationPit))
            new_state.append(get_my_units_amount(obs, units.Zerg.LurkerDen))
            new_state.append(get_my_units_amount(obs, units.Zerg.NydusNetwork))
            new_state.append(get_my_units_amount(obs, units.Zerg.RoachWarren))
            new_state.append(get_my_units_amount(obs, units.Zerg.SpawningPool))
            new_state.append(get_my_units_amount(obs, units.Zerg.SpineCrawler))
            new_state.append(get_my_units_amount(obs, units.Zerg.Spire))
            new_state.append(get_my_units_amount(obs, units.Zerg.SporeCrawler))


        m0 = obs.feature_minimap[0]     # Feature layer of the map's terrain (elevation and shape)
        m0 = m0/10

        m1 = obs.feature_minimap[2]     # Feature layer of creep in the minimap (generally will be quite empty, especially on games without zergs hehe)
        m1[m1 == 1] = 16                # Transforming creep info from 1 to 16 in the map (makes it more visible)

        m2 = obs.feature_minimap[4]     # Feature layer of all visible units (neutral, friendly and enemy) on the minimap
        m2[m2 == 1] = 128               # Transforming own units from 1 to 128 for visibility
        m2[m2 == 3] = 64                # Transforming enemy units from 3 to 64 for visibility
        m2[m2 == 2] = 64
        m2[m2 == 16] = 32               # Transforming mineral fields and geysers from 16 to 32 for visibility

        
        combined_minimap = np.where(m2 != 0, m2, m1)                                    # Overlaying the m2(units) map over the m1(creep) map
        combined_minimap = np.where(combined_minimap != 0, combined_minimap, m0)        # Overlaying the combined m1 and m2 map over the m0(terrain) map
        # combined_minimap = m0+m1+m2
        combined_minimap = trim_feature_minimap(combined_minimap)
        combined_minimap = np.array(combined_minimap)                                   # Tranforming combined_minimap into a np array so tensor flow can interpret it
        combined_minimap = combined_minimap/combined_minimap.max()                      # Normalizing between 0 and 1

        # Lowering the featuremap's resolution
        lowered_minimap = lower_featuremap_resolution(combined_minimap, self.reduction_factor)
        
        # Rotating observation depending on Agent's location on the map so that we get a consistent, generalized, observation
        if not self.base_top_left: 
            lowered_minimap = np.rot90(lowered_minimap, 2)
        
        new_state.extend(lowered_minimap.flatten())   
        final_state = np.array(new_state)
        final_state = np.expand_dims(final_state, axis=0)


        # Displaying the agent's vision in a heatmap using matplotlib every 200 steps (just for debug purpuses, probably will be removed later)
        # if (obs.game_loop[0]/16)%50 == 0:
        #     # Displaying Agent's vision
        #     plt.figure()
        #     plt.imshow(lowered_minimap)
            
        #     plt.show()

        return final_state


    def get_state_dim(self):
        return self._state_size


class Simple64GridState(StateBuilder):
    def __init__(self, grid_size=4):

        self.grid_size = grid_size
        self._state_size = int(19 + 2*(self.grid_size**2))
        self.player_race = 0
        self.base_top_left = None

    def build_state(self, obs):
        if obs.game_loop[0] < 80 and self.base_top_left == None:

            commandcenter = get_units_by_type(obs, units.Terran.CommandCenter)
            nexus = get_units_by_type(obs, units.Protoss.Nexus)
            hatchery = get_units_by_type(obs, units.Zerg.Hatchery)
            if len(commandcenter)>0: 
                townhall = commandcenter[0]
                self.player_race = sc2_env.Race.terran
            if len(nexus)>0:
                townhall = nexus[0]
                self.player_race = sc2_env.Race.protoss
            if len(hatchery)>0:
                townhall = hatchery[0]
                self.player_race = sc2_env.Race.zerg

            self.base_top_left = (townhall.x < 32)

        new_state = []
        new_state.append(obs.player.minerals/6000)
        new_state.append(obs.player.vespene/6000)
        new_state.append(obs.player.food_cap/200)
        new_state.append(obs.player.food_used/200)
        new_state.append(obs.player.food_army/200)
        new_state.append(obs.player.food_workers/200)
        new_state.append((obs.player.food_cap - obs.player.food_used)/200)
        new_state.append(obs.player.army_count/200)
        new_state.append(obs.player.idle_worker_count/200)

        if self.player_race == sc2_env.Race.terran:
            new_state.append(get_my_units_amount(obs, units.Terran.CommandCenter)+
                            get_my_units_amount(obs, units.Terran.OrbitalCommand)+
                            get_my_units_amount(obs, units.Terran.PlanetaryFortress)/2)
            new_state.append(get_my_units_amount(obs, units.Terran.SupplyDepot)/18)
            new_state.append(get_my_units_amount(obs, units.Terran.Refinery)/4)
            new_state.append(get_my_units_amount(obs, units.Terran.EngineeringBay))
            new_state.append(get_my_units_amount(obs, units.Terran.Armory))
            new_state.append(get_my_units_amount(obs, units.Terran.MissileTurret)/4)
            #new_state.append(get_my_units_amount(obs, units.Terran.SensorTower)/1)
            #new_state.append(get_my_units_amount(obs, units.Terran.Bunker)/4)
            new_state.append(get_my_units_amount(obs, units.Terran.FusionCore))
            #new_state.append(get_my_units_amount(obs, units.Terran.GhostAcademy))
            new_state.append(get_my_units_amount(obs, units.Terran.Barracks)/3)
            new_state.append(get_my_units_amount(obs, units.Terran.Factory)/2)
            new_state.append(get_my_units_amount(obs, units.Terran.Starport)/2)

        elif self.player_race == sc2_env.Race.protoss:
            new_state.append(get_my_units_amount(obs, units.Protoss.Nexus))
            new_state.append(get_my_units_amount(obs, units.Protoss.Pylon))
            new_state.append(get_my_units_amount(obs, units.Protoss.Assimilator))
            new_state.append(get_my_units_amount(obs, units.Protoss.Forge))
            new_state.append(get_my_units_amount(obs, units.Protoss.Gateway))
            new_state.append(get_my_units_amount(obs, units.Protoss.CyberneticsCore))
            new_state.append(get_my_units_amount(obs, units.Protoss.PhotonCannon))
            new_state.append(get_my_units_amount(obs, units.Protoss.RoboticsFacility))
            new_state.append(get_my_units_amount(obs, units.Protoss.Stargate))
            new_state.append(get_my_units_amount(obs, units.Protoss.TwilightCouncil))
            new_state.append(get_my_units_amount(obs, units.Protoss.RoboticsBay))
            new_state.append(get_my_units_amount(obs, units.Protoss.TemplarArchive))
            new_state.append(get_my_units_amount(obs, units.Protoss.DarkShrine))
            
        elif self.player_race == sc2_env.Race.zerg:
            new_state.append(get_my_units_amount(obs, units.Zerg.BanelingNest))
            new_state.append(get_my_units_amount(obs, units.Zerg.EvolutionChamber))
            new_state.append(get_my_units_amount(obs, units.Zerg.Extractor))
            new_state.append(get_my_units_amount(obs, units.Zerg.Hatchery))
            new_state.append(get_my_units_amount(obs, units.Zerg.HydraliskDen))
            new_state.append(get_my_units_amount(obs, units.Zerg.InfestationPit))
            new_state.append(get_my_units_amount(obs, units.Zerg.LurkerDen))
            new_state.append(get_my_units_amount(obs, units.Zerg.NydusNetwork))
            new_state.append(get_my_units_amount(obs, units.Zerg.RoachWarren))
            new_state.append(get_my_units_amount(obs, units.Zerg.SpawningPool))
            new_state.append(get_my_units_amount(obs, units.Zerg.SpineCrawler))
            new_state.append(get_my_units_amount(obs, units.Zerg.Spire))
            new_state.append(get_my_units_amount(obs, units.Zerg.SporeCrawler))     

        # Insteading of making a vector for all coordnates on the map, we'll discretize our enemy space
        # and use a 4x4 grid to store enemy positions by marking a square as 1 if there's any enemy on it.

        enemy_grid = np.zeros((self.grid_size,self.grid_size))
        player_grid = np.zeros((self.grid_size,self.grid_size))

        enemy_units = [unit for unit in obs.raw_units if unit.alliance == features.PlayerRelative.ENEMY]
        player_units = [unit for unit in obs.raw_units if unit.alliance == features.PlayerRelative.SELF]
        
        for i in range(0, len(enemy_units)):
            y = int(math.ceil((enemy_units[i].x + 1) / 64/self.grid_size))
            x = int(math.ceil((enemy_units[i].y + 1) / 64/self.grid_size))
            enemy_grid[x-1][y-1] += 1

        for i in range(0, len(player_units)):
            y = int(math.ceil((player_units[i].x + 1) / (64/self.grid_size)))
            x = int(math.ceil((player_units[i].y + 1) / (64/self.grid_size)))
            player_grid[x-1][y-1] += 1

        if not self.base_top_left:
            enemy_grid = np.rot90(enemy_grid, 2)
            player_grid = np.rot90(player_grid, 2)

        # Normalizing the values to always be between 0 and 1 (since the max amount of units in SC2 is 200)
        enemy_grid = enemy_grid/200
        player_grid = player_grid/200
        
        new_state.extend(enemy_grid.flatten())
        new_state.extend(player_grid.flatten())
        final_state = np.expand_dims(new_state, axis=0)
        return final_state

    def get_state_dim(self):
        return self._state_size


class Simple64GridState_SimpleTerran(StateBuilder):
    def __init__(self, grid_size=4):

        self.grid_size = grid_size
        self._state_size = int(12 + 2*(self.grid_size**2))

    def build_state(self, obs):

        new_state = []
        new_state.append(obs.player.minerals/6000)
        new_state.append(obs.player.vespene/6000)
        new_state.append(obs.player.food_cap/200)
        new_state.append(obs.player.food_used/200)
        new_state.append(obs.player.food_army/200)
        new_state.append(obs.player.idle_worker_count/200)

        new_state.append(get_my_units_amount(obs, units.Terran.CommandCenter)+
                        get_my_units_amount(obs, units.Terran.OrbitalCommand)+
                        get_my_units_amount(obs, units.Terran.PlanetaryFortress)/10)
        new_state.append(get_my_units_amount(obs, units.Terran.SupplyDepot)/10)
        new_state.append(get_my_units_amount(obs, units.Terran.Refinery)/10)
        new_state.append(get_my_units_amount(obs, units.Terran.Barracks)/10)
        new_state.append(get_my_units_amount(obs, units.Terran.Factory)/10)
        new_state.append(get_my_units_amount(obs, units.Terran.Starport)/10)  


        enemy_grid = np.zeros((self.grid_size,self.grid_size))
        player_grid = np.zeros((self.grid_size,self.grid_size))

        enemy_units = [unit for unit in obs.raw_units if unit.alliance == features.PlayerRelative.ENEMY]
        player_units = [unit for unit in obs.raw_units if unit.alliance == features.PlayerRelative.SELF]
        
        for i in range(0, len(enemy_units)):
            y = int(math.ceil((enemy_units[i].x + 1) / 64/self.grid_size))
            x = int(math.ceil((enemy_units[i].y + 1) / 64/self.grid_size))
            enemy_grid[x-1][y-1] += 1

        for i in range(0, len(player_units)):
            y = int(math.ceil((player_units[i].x + 1) / (64/self.grid_size)))
            x = int(math.ceil((player_units[i].y + 1) / (64/self.grid_size)))
            player_grid[x-1][y-1] += 1

        # Normalizing the values to always be between 0 and 1 (since the max amount of units in SC2 is 200)
        enemy_grid = enemy_grid/200
        player_grid = player_grid/200
        
        new_state.extend(enemy_grid.flatten())
        new_state.extend(player_grid.flatten())
        final_state = np.array(new_state)
        final_state = np.expand_dims(new_state, axis=0)
        return final_state

    def get_state_dim(self):
        return self._state_size


class SimpleCroppedGridState(StateBuilder):
    def __init__(self, x1=12, y1=12, x2=50, y2=50, grid_size=4, r_enemy=True, r_player=True, r_neutral=False):
        self.grid_size = grid_size
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.r_enemy = r_enemy
        self.r_player = r_player
        self.r_neutral = r_neutral

        num_grids = int(r_enemy) + int(r_player) + int(r_neutral)
        self._state_size = int(num_grids*(self.grid_size**2))

    def build_state(self, obs):
        state = build_cropped_gridstate(obs, self.grid_size, self.x1, self.y1, self.x2, self.y2, self.r_enemy, self.r_player, self.r_neutral)
        return state

    def get_state_dim(self):
        return self._state_size


class UnitStackingState(StateBuilder):
    """
    This state builder creates an input matrix with dimensions (?,4), where the first few lines of the input matrix are
    non-spatial features (minerals, vespene, idle worker count, game alerts, max supply etc) and the remaining lines are
    divided into two groups of unit features: one group for the player, and another for the enemy.

    The two groups of unit features are further divided into two groups: the group of units we only want to know the amount of, 
    and the group of units we want to know the amount, avg health ratio and avg position. The first group is called "amount_unit_types"
    while the second is called "spatial_unit_types". "amount_unit_types" is defined as a list of PySC2 unit types, while
    "spatial_unit_types" is defined as a list of lists of PySC2 unit types, both defined in the __init__() method. 
    
    "spatial_unit_types"is defined as a list of lists so that we can group certain types of units together and treat them like one type. 
    For example, we could group units based on their traversal medium (separate ground from air units), or we could group them based on 
    their attacking capabilities (all air-attacking units together, etc), or we could group them using both criteria (all ground units 
    with ground and air attacking capabilities together, etc). There is really no end to combinations here, and each developer can 
    choose how to implement their version of this class to achieve their desired observation grouping.

    After both the player and enemy units are grouped, and their propeties calculated, we flatten out the output, so that a
    dense neural network layer can use this state matrix as input.
    """
    def __init__(self):
        self.amount_unit_types = []
        self.spatial_unit_types = []
        self._state_size = 0

    def build_state(self, obs):
        '''
        state shape is defined as follows (3,4) for non-spatial features, like minerals, gas, supply, etc,
        
        then (a,4) for amount_unit_types, where "a" is the size of amount_unit_types divided by 4.
        If amount_unit_types has 12 unit types, we need 3 matrix lines to store them, so "a" will be 3.
        
        and then (b,4), where b is the size of spatial_unit_types.

        at the end "state" is flattened to be used by dense input layers.
        '''
        state = np.zeros((3, 4))

        state[0][0] = obs.player.minerals/10000
        state[0][1] = obs.player.vespene/10000
        state[0][2] = obs.player.food_cap/200
        state[0][3] = obs.player.food_used/200
        
        state[1][0] = obs.player.food_army/200
        state[1][1] = obs.player.food_workers/200
        state[1][2] = obs.player.idle_worker_count/20
        state[1][3] = obs.player.army_count/200

        state[2][0] = obs.alerts[0] if len(obs.alerts)>0 else 0
        state[2][1] = obs.action_result[0] if len(obs.action_result)>0 else 0
        state[2][2] = (obs.game_loop[-1]/obs.step_mul)/2000                     #find a way of getting max_steps_training here?
        state[2][3] = get_fog_of_war_percentage(obs)

        # addind amount of each unit in amount_unit_types for the player to state
        state = build_unit_amount_matrix(obs, features.PlayerRelative.SELF, state, self.amount_unit_types, 4, 20)

        # creating matrix of spatial features for the player's unit groups
        state = build_unit_feature_matrix(obs, features.PlayerRelative.SELF, state, self.spatial_unit_types, 200)

        # addind amount of each unit in amount_unit_types for the enemy to state
        state = build_unit_amount_matrix(obs, features.PlayerRelative.ENEMY, state, self.amount_unit_types, 4, 20)

        # creating matrix of spatial features for the enemy's unit groups
        state = build_unit_feature_matrix(obs, features.PlayerRelative.ENEMY, state, self.spatial_unit_types, 200)

        flat_state = state.flatten()
        final_state = np.expand_dims(flat_state, axis=0)
        return final_state

    def get_state_dim(self):
        return self._state_size

class TVTUnitStackingState(UnitStackingState):
    """
    A version of the generic unit stacking state designed for Terran vs Terran gameplay
    """
    def __init__(self):
        self.amount_unit_types = [
            units.Terran.Barracks,
            units.Terran.BarracksTechLab,
            units.Terran.CommandCenter,
            units.Terran.EngineeringBay,
            units.Terran.Factory,
            units.Terran.FactoryTechLab,
            units.Terran.Refinery,
            units.Terran.Starport,
            units.Terran.StarportTechLab,
            units.Terran.SupplyDepot,
            units.Terran.Armory,
            units.Terran.FusionCore,
        ]
        self.spatial_unit_types = [
            # Ground -> Ground/Air
            [units.Terran.Marine, units.Terran.Cyclone, units.Terran.WidowMine],

            # Ground -> Ground
            [units.Terran.Reaper, units.Terran.Marauder, units.Terran.SiegeTank, units.Terran.Hellion, units.Terran.Hellbat],

            # Air -> Ground/Air
            [units.Terran.VikingFighter],

            # Air -> Ground
            [units.Terran.Banshee, units.Terran.LiberatorAG],
            
            # Air -> Air
            [units.Terran.Liberator],

            # Medivac
            [units.Terran.Medivac],

            # Thor
            [units.Terran.Thor],

            # Raven
            [units.Terran.Raven],

            # Battlecruiser
            [units.Terran.Battlecruiser],

            #SCVs
            [units.Terran.SCV],
        ]
        self._state_size = 4 * (3 + 2 * (  math.ceil(len(self.amount_unit_types)/4) + len(self.spatial_unit_types)) )

class TVTUnitStackingEnemyGridState(UnitStackingState):
    """
    A version of the generic unit stacking state designed for SimpleMOTerranWrapper
    In this class we also use a grid for enemy units (to help the agent locate enemy buildings on map)
    """
    def __init__(self, grid_size=4):
        self.grid_size = grid_size
        self.amount_unit_types = [
            units.Terran.Barracks,
            units.Terran.CommandCenter,
            units.Terran.EngineeringBay,
            units.Terran.Factory,
            units.Terran.Refinery,
            units.Terran.Starport,
            units.Terran.Armory,
            units.Terran.FusionCore,
        ]
        self.spatial_unit_types = [
            # Ground -> Ground/Air
            [units.Terran.Marine, units.Terran.Cyclone],

            # Ground -> Ground
            [units.Terran.Reaper, units.Terran.Marauder, units.Terran.SiegeTank, units.Terran.Hellion, units.Terran.Hellbat],

            # Air Units
            [units.Terran.VikingFighter, units.Terran.Liberator, units.Terran.Banshee, units.Terran.LiberatorAG, units.Terran.Raven],

            # Medivac
            [units.Terran.Medivac],

            # Thor
            [units.Terran.Thor],

            # Battlecruiser
            [units.Terran.Battlecruiser],
        ]
        # with current grid_size, amount_unit_types and spatial_unit_types, _state_size should be 92
        self._state_size = 4 * (3 + 2 * (  math.ceil(len(self.amount_unit_types)/4) + len(self.spatial_unit_types)) ) + self.grid_size*self.grid_size

    def build_state(self, obs):
        state = super().build_state(obs)

        enemy_grid = np.zeros( (self.grid_size, self.grid_size) )

        enemy_units = [unit for unit in obs.raw_units if unit.alliance == features.PlayerRelative.ENEMY]

        # then we add all of them to the proper square in the grid (note that this adds units that have already been added by the first
        # for, since we don't filter out overall_units by unit type)
        for i in range(0, len(enemy_units)):
                y = int(math.ceil((enemy_units[i].x + 1) / (obs.map_size.x/self.grid_size)))
                x = int(math.ceil((enemy_units[i].y + 1) / (obs.map_size.y/self.grid_size)))
                enemy_grid[x-1][y-1] += 1

        enemy_grid = enemy_grid/100

        state = np.squeeze(state)
        state = np.append(state, enemy_grid.flatten(), axis=0)
        final_state = np.expand_dims(state, axis=0)
        return final_state


class MultipleUnitGridState(StateBuilder):
    def __init__(self, grid_size=4):
        self.grid_size = grid_size
        self.list_unit_types = [
            # SCVs
            # [units.Terran.SCV],

            # Unit Prod
            # [units.Terran.Barracks, units.Terran.Factory, units.Terran.Starport, units.Terran.CommandCenter],
            
            # Ground units
            [units.Terran.Marine, units.Terran.Cyclone, units.Terran.WidowMine, units.Terran.Reaper,
             units.Terran.Marauder, units.Terran.SiegeTank, units.Terran.Hellion, units.Terran.Hellbat, 
             units.Terran.Thor],

            # Air units
            [units.Terran.VikingFighter, units.Terran.Banshee, units.Terran.Liberator, 
            units.Terran.Medivac,  units.Terran.Raven, units.Terran.Battlecruiser],

            # Any unit or construction that is not contemplated by the lists above will be grouped together in a final list
            # automatically. So, supply depots, refineries, engineering bays etc will all be grouped together.
        ]
        self._state_size = 20 + (self.grid_size * self.grid_size * 2) * (len(self.list_unit_types) + 1)

    def build_state(self, obs):
        state = np.zeros((0))

        # non-spatial features
        state = np.append(state, obs.player.minerals/10000 )
        state = np.append(state, obs.player.vespene/10000 )
        state = np.append(state, obs.player.food_cap/200 )
        state = np.append(state, obs.player.food_used/200 )

        state = np.append(state, obs.player.food_army/200 )
        state = np.append(state, obs.player.food_workers/200 )
        state = np.append(state, obs.player.idle_worker_count/20 )
        state = np.append(state, obs.player.army_count/200 )

        state = np.append(state, obs.alerts[0] if len(obs.alerts)>0 else 0 )
        state = np.append(state, obs.action_result[0] if len(obs.action_result)>0 else 0 )
        state = np.append(state, (obs.game_loop[-1]/obs.step_mul)/2000 )          #find a way of getting max_steps_training here?
        state = np.append(state, get_fog_of_war_percentage(obs) )

        # amount of buildings
        state = np.append(state, get_my_units_amount(obs, units.Terran.CommandCenter)/2 )
        state = np.append(state, get_my_units_amount(obs, units.Terran.SupplyDepot)/18 )
        state = np.append(state, get_my_units_amount(obs, units.Terran.Armory) )
        state = np.append(state, get_my_units_amount(obs, units.Terran.FusionCore) )
        
        state = np.append(state, get_my_units_amount(obs, units.Terran.Barracks)/3 )
        state = np.append(state, get_my_units_amount(obs, units.Terran.Factory)/2 )
        state = np.append(state, get_my_units_amount(obs, units.Terran.Starport)/2 )
        state = np.append(state, get_my_units_amount(obs, units.Terran.EngineeringBay) )

        # creating grids of spatial features (position of buildings and troops)
        # if grid_size = 4 and len(list_unit_types) = 2, player_grid will be of shape (4,4,3)
        player_grid = np.zeros( (self.grid_size, self.grid_size, len(self.list_unit_types)+1) )
        build_multiple_unit_grid(obs, features.PlayerRelative.SELF, player_grid, self.grid_size, self.list_unit_types)

        enemy_grid = np.zeros( (self.grid_size, self.grid_size, len(self.list_unit_types)+1) )
        build_multiple_unit_grid(obs, features.PlayerRelative.ENEMY, enemy_grid, self.grid_size, self.list_unit_types)

        # dividing by 100 so player_grid has values from 0 to 1 as long as we don't have
        # more than 100 units from a unit group in self.list_unit_types (unlikely scenario)
        player_grid = player_grid/100
        enemy_grid = enemy_grid/100

        state = np.append(state, player_grid.flatten(), axis=0)
        state = np.append(state, enemy_grid.flatten(), axis=0)
        final_state = np.expand_dims(state, axis=0)
        return final_state

    def get_state_dim(self):
        return self._state_size


def build_multiple_unit_grid(obs, player, grid, grid_size, unit_groups):
    for group_i, unit_group in enumerate(unit_groups):
        
        # getting all units which type matches one type in unit_group
        group_units = [unit for unit in obs.raw_units if unit.alliance == player and unit.unit_type in unit_group]

        # adding each unit to the proper square in the grid using group_i as the proper depth
        for i in range(0, len(group_units)):
            y = int(math.ceil((group_units[i].x + 1) / (obs.map_size.x/grid_size)))
            x = int(math.ceil((group_units[i].y + 1) / (obs.map_size.y/grid_size)))
            grid[x-1][y-1][group_i] += 1

    # after adding all units from the types in unit_groups we need to add the remaining units whose types are not in unit_groups
    # we begin by collecting all units
    overall_units = [unit for unit in obs.raw_units if unit.alliance == player]
    overall_i = len(unit_groups)

    # then we add all of them to the proper square in the grid (note that this adds units that have already been added by the first
    # for, since we don't filter out overall_units by unit type)
    for i in range(0, len(overall_units)):
            y = int(math.ceil((overall_units[i].x + 1) / (obs.map_size.x/grid_size)))
            x = int(math.ceil((overall_units[i].y + 1) / (obs.map_size.y/grid_size)))
            grid[x-1][y-1][overall_i] += 1

    # now we are going to do a very simple filter to subtract from the overall amount:
    # we basically go trough each depth in our 3D grid and subtract that from our overall_i depth.
    # so, if in grid 0,0 there are 8 units, but those 8 units are all from types already calculated 
    # by the first for, in the end we want grid 0,0 for the overall_i depth to be 8-8 = 0.
    for i in range(0, grid_size):
        for j in range(0, grid_size):
            for k in range(0, overall_i):
                grid[i][j][overall_i] -= grid[i][j][k]

def build_unit_amount_matrix(obs, player, state, amount_unit_types, n_columns, normalize_value):
    new_line = []
    for i, unit_type in enumerate(amount_unit_types):
        new_line.append(get_unit_amount(obs, unit_type, player)/normalize_value)
        if (i+1) % n_columns == 0:
            state = np.append(state, [new_line], axis=0)
            new_line = []
    return state

def build_unit_feature_matrix(obs, player, state, spatial_unit_types, normalize_value):
    for unit_group in spatial_unit_types:
            unit_amount=0
            health_ratio=0
            x_avg=0
            y_avg=0
            len_unit_group = len(unit_group)

            for unit_type in unit_group:
                units = [unit for unit in obs.raw_units if unit.alliance == player and unit.unit_type == unit_type]
                unit_amount += len(units)

                if unit_amount > 0:
                    health_ratio += sum([unit.health_ratio for unit in units]) / unit_amount
                    x_avg += (sum([unit.x for unit in units]) / unit_amount) / obs.map_size.x
                    y_avg += (sum([unit.y for unit in units]) / unit_amount) / obs.map_size.y

            unit_amount = (unit_amount/len_unit_group)/normalize_value #divide by 200 to normalize
            health_ratio = health_ratio/len_unit_group
            x_avg = x_avg/len_unit_group
            y_avg = y_avg/len_unit_group

            new_line = [[unit_amount, health_ratio, x_avg, y_avg]]
            state = np.append(state, new_line, axis=0)
    return state

def trim_feature_minimap(feature_minimap):
    feature_minimap = np.delete(feature_minimap, np.s_[0:12:1], 0)
    feature_minimap = np.delete(feature_minimap, np.s_[44:63:1], 0)
    feature_minimap = np.delete(feature_minimap, np.s_[0:8:1], 1)
    feature_minimap = np.delete(feature_minimap, np.s_[44:63:1], 1)
    return feature_minimap

def build_cropped_gridstate(obs, grid_size, x1, y1, x2, y2, r_enemy:bool, r_player:bool, r_neutral:bool):
    """
    This function generates a series of grids based on a cropped raw SC2 representation.
    You can have a return vector with only the enemy, player or neutral units, or any mix of these three.
    The cropping logic is the following: you could have a 64x64 SC2 map but the playable part is restricted 
    to a rectangle defined by the top-left point with x1=20 and y1=25 and the bottom right point with x2=40, y2=45.
    The points (x1,y1) and (x2,y2) will be used to calculate the position of units relative to the playable area.
    This would, for example, mean that a unit that originally was on the (24,27) point will now be on the (4,2) on our cropped representation.

    Args:
        obs: raw SC2 observation
        grid_size (int): 
        x1 (int): is the x position of the top-left point of the rectangle that you want to crop the map
        y1 (int): is the y positiono f the top-left point of the rectangle that you want to crop the map
        r_enemy (bool): whether or not the grid with enemy units will be returned
        r_player (bool): whether or not the grid with player units will be returned
        r_neutral (bool): whether or not the grid with neutral units will be returned

    Returns:
        A list with the flattened grids from the enemy, player and neutral units depending on the input arguments.
    """
    new_state = []

    cropped_x = x2-x1
    cropped_y = y2-y1

    if r_enemy:
        enemy_grid = np.zeros((grid_size,grid_size))
        enemy_units = [unit for unit in obs.raw_units if unit.alliance == features.PlayerRelative.ENEMY]

        build_cropped_grid(enemy_units, cropped_x, cropped_y, grid_size, enemy_grid, x1, y1)

        # Normalizing the values to always be between 0 and 1 (since the max amount of units in SC2 is 200)
        # This code line can be commented out depending on your desired representation
        enemy_grid = enemy_grid/200
        # Adding the flattened grid matrix to the new_state
        new_state.extend(enemy_grid.flatten())

    if r_player:
        player_grid = np.zeros((grid_size,grid_size))
        player_units = [unit for unit in obs.raw_units if unit.alliance == features.PlayerRelative.SELF]

        build_cropped_grid(player_units, cropped_x, cropped_y, grid_size, player_grid, x1, y1)

        # Normalizing the values to always be between 0 and 1 (since the max amount of units in SC2 is 200)
        # This code line can be commented out depending on your desired representation
        player_grid = player_grid/200
        # Adding the flattened grid matrix to the new_state
        new_state.extend(player_grid.flatten())

    if r_neutral:
        neutral_grid = np.zeros((grid_size,grid_size))
        neutral_units = [unit for unit in obs.raw_units if unit.alliance == features.PlayerRelative.NEUTRAL]

        build_cropped_grid(neutral_units, cropped_x, cropped_y, grid_size, neutral_grid, x1, y1)

        # Normalizing the values to always be between 0 and 1 (since the max amount of units in SC2 is 200)
        # This code line can be commented out depending on your desired representation
        neutral_grid = neutral_grid/200
        # Adding the flattened grid matrix to the new_state
        new_state.extend(neutral_grid.flatten())

    final_state = np.array(new_state)
    final_state = np.expand_dims(new_state, axis=0)
    return final_state

def build_cropped_grid(unit_list, cropped_x, cropped_y, grid_size, unit_grid, x1, y1):
    for i in range(0, len(unit_list)):
        unit_x = unit_list[i].x - x1
        unit_y = unit_list[i].y - y1

        if( (unit_x < cropped_x) and (unit_y < cropped_y)):
            y = int(math.ceil( (unit_x + 1) / (cropped_x/grid_size) ))
            x = int(math.ceil( (unit_y + 1) / (cropped_y/grid_size) ))
            unit_grid[x-1][y-1] += 1
