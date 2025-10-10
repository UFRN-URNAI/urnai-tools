import os
import sys

from absl import app
from pysc2.env import sc2_env

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from urnai.sc2.actions.collectables import CollectablesActionSpace
from urnai.sc2.environments.sc2environment import SC2Env
from urnai.sc2.models.atarinet_model import AtariNetModel
from urnai.sc2.rewards.collectables import CollectablesReward
from urnai.sc2.states.deepmind_state import DeepmindState
from urnai.logging.wandb_logger import WandbLogger
from experiments.solves.solve_collectables_sb3 import declare_wandb_run

def make_env():

    class QuickEnv:
        def __init__(self):
            players = [sc2_env.Agent(sc2_env.Race.terran)]
            self._env = SC2Env(map_name='CollectMineralShards', visualize=False, 
                step_mul=16, players=players)

            self._state = DeepmindState()
            self._action_space = CollectablesActionSpace()
            self._reward = CollectablesReward()

            self._obs = self._env.reset()
        
        def step(self, action):
            action = self._action_space.get_action(action, self._obs)

            obs, reward, terminated, truncated = self._env.step(action)

            self._obs = obs
            obs = self._state.update(self._obs)
            reward = self._reward.get(self._obs, reward, terminated, truncated)
            return obs, reward, terminated or truncated
        
        def reset(self):
            self._obs = self._env.reset()

    env = QuickEnv()
    return env

def train(model):

    env = make_env()
    state = env._state.update(env._obs)
    logger = WandbLogger()
    total_reward = 0

    for ep in range(100):
        for step in range(300 * 100):
            function_id, args = model.predict(state)
            action = function_id #TODO: Make ActionSpace receive action arguments

            next_state, reward, done = env.step(action)
            total_reward += reward

            model.learn(state, action, reward, next_state, done)

            state = next_state

            if done:
                break

        print(step, total_reward, total_reward/(step + 1))

        logger.log({
            "mean loss": model.total_loss / step,
            "reward per ep" : total_reward,
            "ep length" : step,
            "episode" : ep
        })

        total_reward = 0
        model.total_loss = 0
        env.reset()

def main(_):

    config_dict = {
        "model_save_name": "Atarinet-test"
    }
    wandb_run = declare_wandb_run(config_dict)

    model = AtariNetModel(map_name='CollectMineralShards')
    
    train(model)

    wandb_run.finish()

if __name__ == '__main__':
    app.run(main)