"""
Outdated file
"""

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent.parent))

from absl import app
from urnai.envs.vizdoom import VizdoomEnv
from urnai.trainers.trainer import Trainer
from urnai.agents.generic_agent import GenericAgent
from urnai.agents.actions.vizdoom_wrapper import VizdoomHealthGatheringWrapper
from urnai.agents.rewards.vizdoom import VizDoomHealthGatheringReward  
from urnai.agents.states.vizdoom import VizDoomHealthGatheringState
from urnai.models.ddqn_keras import DDQNKeras 
from urnai.models.dqn_keras_mem import DQNKerasMem
from urnai.models.model_builder import ModelBuilder
from urnai.utils.tf_utils import ignore_tensorflow_gpu, allow_memory_growth 
from datetime import datetime

#force tf cpu if using tf_gpu
#tf_utils.ignore_tensorflow_gpu()

#force tf_gpu to allow more memory usage
#uncomment only if needed
#tf_utils.allow_memory_growth()

def main(unused_argv):
    try:
        env = VizdoomEnv(parentdir + os.path.sep +"utils/vizdoomwads/health_gathering.wad", render=False, doommap=None, res=VizdoomEnv.RES_640X480)

        training_date = str(datetime.now()).replace(" ","_").replace(":","_").replace(".","_")

        action_wrapper = VizdoomHealthGatheringWrapper()
        state_builder = VizDoomHealthGatheringState(env.get_screen_width(), env.get_screen_height(), slices=3)

        helper = ModelBuilder()
        helper.add_convolutional_layer(filters=32, input_shape=(env.get_screen_height(), env.get_screen_width(), 1)) #1 means grayscale images 
        helper.add_convolutional_layer(filters=16)
        helper.add_fullyconn_layer(50)
        helper.add_output_layer(action_wrapper.get_action_space_dim())
        #dq_network = DDQNKeras(action_wrapper=action_wrapper, state_builder=state_builder, learning_rate=0.005, gamma=0.90, use_memory=False, per_episode_epsilon_decay = True, build_model=helper.get_model_layout())
        dq_network = DQNKerasMem(action_wrapper=action_wrapper, state_builder=state_builder, learning_rate=0.005, gamma=0.90, use_memory=False, per_episode_epsilon_decay = True, build_model=helper.get_model_layout())
        agent = GenericAgent(dq_network, VizDoomHealthGatheringReward(method="cumulative"))
        #trainer = Trainer(env, agent, file_name=training_date + os.path.sep + "frozenlake_test_dqnKeras_kerasmem_3000ep", save_every=100, enable_save=True)
        trainer = Trainer(env, agent, save_path="/Users/marcocspc/Downloads/", file_name="vizdoom_jsontrainer_3000ep", save_every=1000, enable_save=True)
        env = VizdoomEnv(parentdir + os.path.sep +"utils/vizdoomwads/health_gathering.wad", render=True, doommap=None, res=VizdoomEnv.RES_160X120)
        trainer.env = env

        #trainer.train(num_episodes=3000, reward_from_env=True, max_steps=500)
        trainer.play(num_matches=100)

    except KeyboardInterrupt:
        pass

if __name__ == '__main__':
    app.run(main)
