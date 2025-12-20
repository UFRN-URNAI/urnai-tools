import os
import numpy as np
import sys

from absl import app

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from urnai.sc2.models.atarinet_model import AtariNetModel
from urnai.logging.wandb_logger import WandbLogger
from experiments.atarinet.quickpersistence import QuickPersistence
from experiments.atarinet.quickenv import QuickEnv
from experiments.atarinet.quicklog import QuickLog
from experiments.solves.solve_collectables_sb3 import declare_wandb_run
from experiments.atarinet.test_nn_debug import screenshot, save_in_file, load_from_file
from experiments.atarinet.simple_env import SimpleRFEnv

N_STEPS = 120
TRAINING = False
WANDB = False
N_EPS = 1

def run(model : AtariNetModel, persistence : QuickPersistence):

    model.training = TRAINING

    # TODO: temporary
    adapt_state = lambda state : [state["screen"], state["minimap"], state["nonspatial"]]

    logger = QuickLog()
    #env = QuickEnv()
    env = SimpleRFEnv()
    #state = model.make_frame_stack(env._state.update(env._obs))
    obs, _ = env.reset()
    state = model.make_frame_stack(adapt_state(obs))
    
    if WANDB: wandb_logger = WandbLogger()

    for ep in range(0, N_EPS):
        for step in range(N_STEPS): 
            function_id, args = model.predict(state)
            action = function_id #TODO: Make ActionSpace receive action arguments

            next_state, reward, done, _, _ = env.step(action)
            next_state = model.make_frame_stack(adapt_state(next_state))
            #save_in_file(next_state, step, config_dict)
            #load_from_file(0, config_dict)
            #screenshot(print_list, step)
            import cv2
            cv2.imwrite(f"data/simple_env/frame_{step}.png", env.unwrapped.render_rgb())
            print(action)

            if done:
                next_state = None

            if TRAINING: model.learn(state, action, reward, next_state, done)
            state = next_state

            logger.log_during_ep(action, reward)

            if done:
                break

        if WANDB: logger.wandb_log(model, step, ep, wandb_logger)

        logger.end_of_ep()
        model.new_ep()
        print(f"end of ep {ep}")

        #env.reset()
        #state = model.make_frame_stack(env._state.update(env._obs))
        obs, _ = env.reset()
        state = model.make_frame_stack(adapt_state(obs))

        persistence.save(ep, eps_until_save = 200)

def main(_):
    try:
        config_dict = {
            "model_save_name": "Atarinet-test-v6"
        }
        if WANDB: wandb_run = declare_wandb_run(config_dict)

        model = AtariNetModel(map_name='CollectMineralShards')
        path = f"saves/models/{config_dict['model_save_name']}/"

        persistence = QuickPersistence(path, model, config_dict)
        persistence.load(ep = 2999)
        
        run(model, persistence)

        if WANDB: wandb_run.finish()
        
    except KeyboardInterrupt:
        print("Training interrupted by user")

if __name__ == '__main__':
    app.run(main)
