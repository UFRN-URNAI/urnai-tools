import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent.parent))

import wandb
import datetime

from absl import app
from urnai.envs.clickgame import ClickGameEnv
from urnai.trainers.trainer import Trainer
from urnai.agents.generic_agent import GenericAgent
from urnai.agents.rewards.default import PureReward
from urnai.agents.states.gym import FlatState
from urnai.agents.actions.clickgame_wrapper import ClickGameWrapper, ClickGameBDQWrapper
from urnai.models.ddqn_keras_mo import DDQNKerasMO
from urnai.models.BDQ import BDQ
from urnai.models.model_builder import ModelBuilder

from urnai.utils.reporter import Reporter as rp

def declare_trainer(): 
    env = ClickGameEnv(render=False, board_shape=[10, 10], max_steps=200)

    action_wrapper = ClickGameBDQWrapper(10, 10)
    state_builder = FlatState(env.board_shape[0]*env.board_shape[1])
    
    dq_network = BDQ(action_wrapper=action_wrapper, state_builder=state_builder, per_episode_epsilon_decay=True,
        lr=0.0001, epsilon_decay=0.992, epsilon_min=0, memory_maxlen=50000, min_memory_size=128, batch_size=32)

    agent = GenericAgent(dq_network, PureReward())

    trainer = Trainer(env, agent, save_path='../agentes-treinados-tcc/click-game', file_name="BDQ-01-sigmoid",
                    save_every=50, enable_save=True, relative_path=True, reset_epsilon=False,
                    max_training_episodes=400, max_steps_training=300,
                    max_test_episodes=100, max_steps_testing=300, rolling_avg_window_size=20)
    
    wandb.init(project="clickgame", name=trainer.file_name, entity="lpdcalves")
    
    return trainer

def main(unused_argv):
    try:
        trainer = declare_trainer()
        trainer.train()
        #trainer.play()

    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    app.run(main)
