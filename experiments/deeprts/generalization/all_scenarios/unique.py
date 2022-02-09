from urnai.scenarios.generalization.rts.collectables import GeneralizedCollectablesScenario 
from urnai.scenarios.generalization.rts.findanddefeat import GeneralizedFindAndDefeatScenario
from urnai.scenarios.generalization.rts.defeatenemies import GeneralizedDefeatEnemiesScenario
from urnai.scenarios.generalization.rts.buildunits import GeneralizedBuildUnitsScenario
import numpy as np
import sys,os
from urnai.utils.numpy_utils import save_iterable_as_csv 

SCENARIO_LIST = [GeneralizedCollectablesScenario, GeneralizedFindAndDefeatScenario, GeneralizedDefeatEnemiesScenario, GeneralizedBuildUnitsScenario]
GAME_LIST = [GeneralizedCollectablesScenario.GAME_DEEP_RTS, GeneralizedCollectablesScenario.GAME_STARCRAFT_II]

episodes = 4
steps = 5
print_collectables_map = False 

#game = Scenario.GAME_STARCRAFT_II
#method = 'single'

for first_game in GAME_LIST:
    game = first_game 
    method = 'multiple'
    PRINT_MAP = False
    SAVE_MAP = True 
    env_state = False
    env_reward = False
    for Scenario in SCENARIO_LIST:
        env = Scenario(game = game, render = False, fit_to_screen=True, method=method)
        action_wrapper = env.get_default_action_wrapper()
        state_builder = env.get_default_state_builder()
        reward_builder = env.get_default_reward_builder()
        np.set_printoptions(threshold=sys.maxsize)

        text = "Choose:\n"
        cont = 0

        print("Running Scenario {}".format(env.__class__.__name__))

        for action in action_wrapper.get_actions():
            idx = action_wrapper.get_actions().index(action) 
            text += "\t{} - {}\n".format(idx, action_wrapper.get_action_name_str_by_int(action))

        for ep in range(episodes):

            reward = 0
            done = False
            state = env.reset()
            action_wrapper.reset()
            print("Episode " + str(ep + 1))
            print("Environment now is: " + str(env.game))
            print("Action Wrapper is " + action_wrapper.action_wrapper.__class__.__name__)
            print("StateBuilder is " + state_builder.state_builder.__class__.__name__)
            print("RewardBuilder is " + reward_builder.reward_builder.__class__.__name__)
            total_ep_reward = 0
            
            for step in range(steps):
                    print("Step " + str(step + 1))

                    print(env.env.render)
                    
                    action = None

                    #try:
                    #    action = int(input(text))
                    #except ValueError:
                    #    #No-action
                    #    action = action_wrapper.get_no_action()

                    action = action_wrapper.get_no_action()
                    if state is not None:
                        action = action_wrapper.get_action(action, state)
                    else:
                        action = action_wrapper.get_no_action()

                    state, reward, done = env.step(action)
                    if not env_reward: reward = reward_builder.get_reward(state)
                    if not env_state: state = state_builder.build_state(state) 
                    total_ep_reward += reward

                    if env.game == Scenario.GAME_DEEP_RTS:
                        for player in env.env.game.players:
                            idx = player.get_id() 
                            player_stats = '''Player {player} stats: 
            Oil: {o}
            Gold: {g}
            Lumber: {l}
            Food: {f}
            Number of archers: {na}'''.format(player=idx+1,o=player.oil,g=player.gold,l=player.lumber,f=player.food,na=player.num_archer)
                            print(player_stats)

                        if print_collectables_map:
                            if 'collectables_map' in env.__dict__.keys():
                                a = env.collectables_map.astype(int)
                                np.savetxt(os.path.expanduser('~') + os.path.sep + 'curr_coll_map.csv', a, fmt='%i',delimiter=",")
                    else:
                        if PRINT_MAP:
                            print("Map shape: {}".format(state.feature_minimap[4].shape))
                            print("Map: {}".format(state.feature_minimap[4]))

                    if SAVE_MAP:
                        if env.game == Scenario.GAME_DEEP_RTS:
                            dire = os.path.expanduser("~") + os.path.sep + "urnai_maps" + os.path.sep + "drts" + os.path.sep + str(step) 
                            try: os.makedirs(dire)
                            except FileExistsError: pass
                            #save_iterable_as_csv(state["state"], directory=dire)
                            save_iterable_as_csv(state, directory=dire, convert_to_int=False)
                        else:
                            dire = os.path.expanduser("~") + os.path.sep + "urnai_maps" + os.path.sep + "sc2" + os.path.sep + str(step) 
                            try: os.makedirs(dire)
                            except FileExistsError: pass
                            #for layer in state.feature_minimap:
                                #save_iterable_as_csv(layer, file_name="layer" + str(state.feature_minimap.index(layer)),directory=dire) 
                            save_iterable_as_csv(state, directory=dire, convert_to_int=False) 

                    print("Reward: {r}".format(r=reward))
                    print("Total Episode Reward: {r}".format(r=total_ep_reward))

                    if done: break
