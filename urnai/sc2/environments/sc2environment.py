from pysc2.env import sc2_env
from pysc2.env.environment import StepType
from pysc2.lib import actions, features

from urnai.environments.environment_base import EnvironmentBase


class SC2Env(EnvironmentBase):
    def __init__(
            self,
            map_name='Simple64',
            player_race=sc2_env.Race.terran,
            enemy_race=sc2_env.Race.random,
            difficulty=sc2_env.Difficulty.very_easy,
            render=False,
            reset_done=True,
            spatial_dim=16,
            step_mul=16,
            game_steps_per_episode=0,
            realtime=False,
            self_play=False,
            action_space=actions.ActionSpace.RAW,
            use_raw_units=True,
            raw_resolution=64,
            use_feature_units=True,
            screen=64,
            minimap=64,
    ):
        super().__init__(map_name, render, reset_done)

        self.self_play = self_play
        self.step_mul = step_mul
        self.game_steps_per_episode = game_steps_per_episode
        self.spatial_dim = spatial_dim
        self.player_race = player_race
        self.enemy_race = enemy_race
        self.difficulty = difficulty
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

        self.start(action_space, use_raw_units, raw_resolution, 
                   use_feature_units, screen, minimap)

    def start(
            self,
            action_space=actions.ActionSpace.RAW,
            use_raw_units=True,
            raw_resolution=64,
            use_feature_units=True,
            screen=64,
            minimap=64,
    ):
        self.done = False
        if self.env_instance is None:
            self.env_instance = sc2_env.SC2Env(
                map_name=self.id,
                visualize=self.render,
                players=self.players,
                agent_interface_format=features.AgentInterfaceFormat(
                    action_space=action_space,
                    use_raw_units=use_raw_units,
                    raw_resolution=raw_resolution,
                    use_feature_units=use_feature_units,
                    feature_dimensions=features.Dimensions(screen, minimap),
                ),
                step_mul=self.step_mul,
                game_steps_per_episode=self.game_steps_per_episode,
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

        obs.step_mul = self.step_mul
        obs.map_size = self.env_instance._interface_formats[0]._raw_resolution

        return obs, reward, done
