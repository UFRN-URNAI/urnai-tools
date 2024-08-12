import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent))

from absl import app
from pysc2.env import sc2_env

from urnai.models.dqn_pytorch import DQNPytorch
from urnai.sc2.actions.collectables import CollectablesActionSpace
from urnai.sc2.agents.sc2_agent import SC2Agent
from urnai.sc2.environments.sc2environment import SC2Env
from urnai.sc2.rewards.collectables import CollectablesReward
from urnai.sc2.states.collectables import CollectablesState
from urnai.trainers.trainer import Trainer


def declare_trainer():
    players = [sc2_env.Agent(sc2_env.Race.terran)]
    env = SC2Env(map_name='CollectMineralShards', visualize=False, 
                 step_mul=16, players=players)
	
	
    action_space = CollectablesActionSpace()
    state_builder = CollectablesState()
    reward_builder = CollectablesReward()

    model = DQNPytorch(action_space, state_builder)

    agent = SC2Agent(action_space, state_builder, model, reward_builder)

    trainer = Trainer(env, agent,
                      max_training_episodes=200, max_steps_training=100000,
                      max_playing_episodes=200, max_steps_playing=100000)
    return trainer

def main(unused_argv):
    try:
        trainer = declare_trainer()
        trainer.train()
        # trainer.play()

    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    app.run(main)