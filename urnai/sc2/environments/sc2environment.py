import sys

from absl import flags
from pysc2.env import sc2_env
from pysc2.env.environment import TimeStep
from pysc2.lib import actions, features

from urnai.environments.environment_base import EnvironmentBase


class SC2Env(EnvironmentBase):
    def __init__(
            self,
            map_name: str = 'Simple64',
            players: list = None,
            visualize: bool = False,
            reset_done: bool = True,
            spatial_dim: int = 16,
            step_mul: int = 16,
            game_steps_per_episode: int = 0,
            realtime: bool = False,
            action_space: actions.ActionSpace = actions.ActionSpace.RAW,
            use_raw_units: bool = True,
            raw_resolution: int = 64,
            use_feature_units: bool = True,
            screen: int = 64,
            minimap: int = 64,
    ):
        super().__init__(map_name, visualize, reset_done)

        # TODO: Investigar como remover essa dependência
        flags.FLAGS(sys.argv)

        self.step_mul = step_mul
        self.game_steps_per_episode = game_steps_per_episode
        self.spatial_dim = spatial_dim
        if(players is None):
            players = [
                sc2_env.Agent(sc2_env.Race.random),
                sc2_env.Bot(sc2_env.Race.random, sc2_env.Difficulty.very_easy),
            ]
        self.players = players
        self.num_agents = sum(1 for p in players if isinstance(p, sc2_env.Agent))
        self.realtime = realtime

        self.start(action_space, use_raw_units, raw_resolution, 
                   use_feature_units, screen, minimap)

    def start(
            self,
            action_space: actions.ActionSpace = actions.ActionSpace.RAW,
            use_raw_units: bool = True,
            raw_resolution: int = 64,
            use_feature_units: bool = True,
            screen: int = 64,
            minimap: int = 64,
    ) -> None:
        self.terminated = False
        self.truncated = False
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

    def step(self, action_list: list) -> tuple[list[list], list[int], bool, bool]:
        timestep = self.env_instance.step(action_list)
        obs, reward, terminated, truncated = self._parse_timestep(timestep)
        self.terminated = terminated
        self.truncated = truncated
        return obs, reward, terminated, truncated

    def reset(self) -> list[list]:
        timestep = self.env_instance.reset()
        obs, *_ = self._parse_timestep(timestep)
        return obs

    def close(self) -> None:
        self.env_instance.close()

    def restart(self) -> None:
        self.close()
        self.reset()

    def _parse_timestep(
            self, timestep: list[TimeStep]
        ) -> tuple[list[list], list[int], bool, bool]:
        """
        Returns a [[Observation], [Reward], Terminated, Truncated] tuple 
        parsed from a given timestep.
        """
        obs = []
        reward = []
        map_size = self.env_instance._interface_formats[0]._raw_resolution
        
        for agent_index in range(self.num_agents):
            obs.append(timestep[agent_index].observation)
            reward.append(timestep[agent_index].reward)
            obs[agent_index].step_mul = self.step_mul
            obs[agent_index].map_size = map_size
        
        terminated = any(o.player_result for o in self.env_instance._obs)
        current_steps = self.env_instance._episode_steps
        limit_steps = self.env_instance._episode_length
        truncated = current_steps >= limit_steps

        return obs, reward, terminated, truncated
