import sys

from absl import flags
from pysc2.env import sc2_env
from pysc2.env.environment import TimeStep
from pysc2.lib import actions, features

from urnai.environments.environment_base import EnvironmentBase


class SC2Env(EnvironmentBase):
    def __init__(
            self,
            map_name='Simple64',
            player_race=sc2_env.Race.random,
            enemy_race=sc2_env.Race.random,
            difficulty=sc2_env.Difficulty.very_easy,
            visualize=False,
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
        super().__init__(map_name, visualize, reset_done)

        FLAGS = flags.FLAGS
        FLAGS(sys.argv)

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
        self.terminated = False
        self.truncated = False

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
    ) -> None:
        self.done = False
        if self.env_instance is None:
            self.env_instance = sc2_env.SC2Env(
                map_name=self.map_name,
                visualize=self.visualize,
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

    def step(self, action) -> tuple[list, int, bool, bool]:
        timestep = self.env_instance.step(action)
        obs, reward, terminated, truncated = self.parse_timestep(timestep)
        self.terminated = terminated
        self.truncated = truncated
        return obs, reward, terminated, truncated

    def reset(self) -> list:
        timestep = self.env_instance.reset()
        obs, reward, terminated, truncated = self.parse_timestep(timestep)
        return obs

    def close(self) -> None:
        self.env_instance.close()

    def restart(self) -> None:
        self.close()
        self.reset()

    def parse_timestep(self, timestep: TimeStep) -> tuple[list, int, bool, bool]:
        """
        Returns a [Observation, Reward, Terminated, Truncated] tuple 
        parsed from a given timestep.
        """
        ts = timestep[0]
        obs, reward = ts.observation, ts.reward
        terminated = any(obs.player_result) # TODO: Check if this works
        current_steps = self.env_instance._episode_steps
        limit_steps = self.env_instance._episode_length
        truncated = current_steps >= limit_steps
        # add step_mul and map_size to obs

        obs.step_mul = self.step_mul
        obs.map_size = self.env_instance._interface_formats[0]._raw_resolution

        return obs, reward, terminated, truncated
