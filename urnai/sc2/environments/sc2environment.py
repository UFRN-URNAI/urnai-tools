from pysc2.env import sc2_env
from pysc2.env.environment import StepType
from pysc2.lib import actions, features

from urnai.environments.environment_base import EnvironmentBase

sc2_races = {
    'terran': sc2_env.Race.terran,
    'protoss': sc2_env.Race.protoss,
    'zerg': sc2_env.Race.zerg,
    'random': sc2_env.Race.random}
sc2_difficulties = {
    'very_easy': sc2_env.Difficulty.very_easy,
    'easy': sc2_env.Difficulty.easy,
    'medium': sc2_env.Difficulty.medium,
    'medium_hard': sc2_env.Difficulty.medium_hard,
    'hard': sc2_env.Difficulty.hard,
    'harder': sc2_env.Difficulty.harder,
    'very_hard': sc2_env.Difficulty.very_hard,
    'cheat_vision': sc2_env.Difficulty.cheat_vision,
    'cheat_money': sc2_env.Difficulty.cheat_money,
    'cheat_insane': sc2_env.Difficulty.cheat_insane}


class SC2Env(EnvironmentBase):
    def __init__(
            self,
            map_name='Simple64',
            players=None,
            player_race='terran',
            enemy_race='random',
            difficulty='very_easy',
            render=False,
            reset_done=True,
            spatial_dim=16,
            step_mul=16,
            game_steps_per_ep=0,
            obs_features=None,
            realtime=False,
            self_play=False,
    ):
        super().__init__(map_name, render, reset_done)

        self.self_play = self_play
        self.step_mul = step_mul
        self.game_steps_per_ep = game_steps_per_ep
        self.spatial_dim = spatial_dim
        self.player_race = get_sc2_race(player_race)
        self.enemy_race = get_sc2_race(enemy_race)
        self.difficulty = get_sc2_difficulty(difficulty)
        if self.self_play:
            self.players = [
                sc2_env.Agent(self.player_race),
                sc2_env.Agent(self.enemy_race),
            ]
        else:
            self.players = [
                sc2_env.Agent(self.player_race),
                sc2_env.Bot(self.enemy_race, self.difficulty),
            ]
        self.realtime = realtime
        self.done = False

        self.start()

    def start(self):
        self.done = False
        if self.env_instance is None:
            self.env_instance = sc2_env.SC2Env(
                map_name=self.id,
                visualize=self.render,
                players=self.players,
                #TODO: It might be a good thing to change the paramaters at will,
                # and don't simply use the default values
                agent_interface_format=features.AgentInterfaceFormat(
                    action_space=actions.ActionSpace.RAW,
                    use_raw_units=True,
                    raw_resolution=64,
                    use_feature_units=True,
                    feature_dimensions=features.Dimensions(screen=64, minimap=64),
                ),
                step_mul=self.step_mul,
                game_steps_per_episode=self.game_steps_per_ep,
                realtime=self.realtime,
            )

    def step(self, action):
        timestep = self.env_instance.step(action)
        obs, reward, done = self.parse_timestep(timestep)
        self.done = done
        return obs, reward, done

    def reset(self):
        timestep = self.env_instance.reset()
        obs, reward, done = self.parse_timestep(timestep)
        return obs

    def close(self):
        self.env_instance.close()

    def restart(self):
        self.close()
        self.reset()

    def parse_timestep(self, timestep):
        """Returns a [Observation, Reward, Done] tuple parsed from a given timestep."""
        ts = timestep[0]
        obs, reward, done = ts.observation, ts.reward, ts.step_type == StepType.LAST
        # add step_mul to obs

        # TODO: Check if this code is right (does "obs" have this members?)
        obs.step_mul = self.step_mul
        obs.map_size = self.env_instance._interface_formats[0]._raw_resolution

        return obs, reward, done

def get_sc2_race(sc2_race: str):
    out = sc2_races.get(sc2_race)
    if out is not None:
        return out
    else:
        raise Exception("Chosen race for StarCraft II doesn't match any known races."
                        " Try: 'terran', 'protoss', 'zerg' or 'random'")


def get_sc2_difficulty(sc2_difficulty: str):
    out = sc2_difficulties.get(sc2_difficulty)
    if out is not None:
        return out
    else:
        raise Exception("Chosen difficulty for StarCraft II doesn't match any known "
                        "difficulties. Try: 'very_easy', 'easy', 'medium', "
                        "'medium_hard', 'hard', 'harder', 'very_hard', "
                        "'cheat_vision', 'cheat_money' or 'cheat_insane'")