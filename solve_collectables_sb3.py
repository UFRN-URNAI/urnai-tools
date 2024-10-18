import os

from gymnasium import spaces
from pysc2.env import sc2_env
from stable_baselines3 import PPO

from urnai.sc2.actions.collectables import CollectablesActionSpace
from urnai.sc2.environments.sc2environment import SC2Env
from urnai.sc2.environments.stablebaselines3.custom_env import CustomEnv
from urnai.sc2.rewards.collectables import CollectablesReward
from urnai.sc2.states.collectables import CollectablesState

players = [sc2_env.Agent(sc2_env.Race.terran)]
env = SC2Env(map_name='CollectMineralShards', visualize=False, 
            step_mul=16, players=players)
state = CollectablesState()
urnai_action_space = CollectablesActionSpace()
reward = CollectablesReward()

# Define action and observation space
action_space = spaces.Discrete(n=4, start=0)
observation_space = spaces.Box(low=0, high=255, shape=(4096, ), dtype=float)

# Create the custom environment
custom_env = CustomEnv(env, state, urnai_action_space, reward, observation_space, 
                       action_space)


# models_dir = "saves/models/DQN"
models_dir = "saves/models/PPO"
logdir = "saves/logs"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(logdir):
    os.makedirs(logdir)

# If training from scratch, uncomment 1
# If loading a model, uncomment 2

## 1 - Train and Save model

# model=DQN("MlpPolicy",custom_env,buffer_size=100000,verbose=1,tensorboard_log=logdir)
model=PPO("MlpPolicy", custom_env, verbose=1, tensorboard_log=logdir)

TIMESTEPS = 10000
for i in range(1,30):
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="DQN")
    model.save(f"{models_dir}/{TIMESTEPS*i}")

## 1 - End

## 2 - Load model
# model = PPO.load(f"{models_dir}/40000.zip", env = custom_env)
## 2 - End

vec_env = model.get_env()
obs = vec_env.reset()

# Test model
for _ in range(10000):
    action, _state = model.predict(obs, deterministic=True)
    # print(action)
    obs, rewards, done, info = vec_env.step(action)

env.close()