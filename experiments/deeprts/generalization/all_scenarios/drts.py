from urnai.scenarios.generalization.rts.defeatenemies import GeneralizedDefeatEnemiesScenario as Scenario
#from urnai.scenarios.generalization.rts.findanddefeat import GeneralizedFindAndDefeatScenario as Scenario
#from urnai.scenarios.generalization.rts.collectables import GeneralizedCollectablesScenario as Scenario
#from urnai.scenarios.generalization.rts.buildunits import GeneralizedBuildUnitsScenario as Scenario
import numpy as np
import sys,os

episodes = 100
steps = 1000
print_collectables_map = False 

env = Scenario(game = Scenario.GAME_DEEP_RTS, render = True, fit_to_screen=True)
action_wrapper = env.get_default_action_wrapper()
np.set_printoptions(threshold=sys.maxsize)

text = "Choose:\n"
cont = 0

print("Running Scenario {}".format(env.__class__.__name__))

for action in action_wrapper.get_actions():
    text += "\t{} - {}\n".format(action, action_wrapper.get_action_name_str_by_int(action))

for ep in range(episodes):
    reward = 0
    done = False
    state = env.reset()
    print("Episode " + str(ep + 1))
    total_ep_reward = 0
    
    for step in range(steps):
            print("Step " + str(step + 1))

            print(env.env.render)
            
            action = None

            try:
                action = int(input(text))
            except ValueError:
                #No-action
                action = action_wrapper.noaction 

            if state is not None:
                action = action_wrapper.get_action(action, state)
            else:
                action = action_wrapper.noaction 

            state, reward, done = env.step(action)
            total_ep_reward += reward

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

            if reward == 0: 
                print("Reward: {r}".format(r=reward))
            else:
                print("YAY!!!!!!!  Reward: {r} !!!!!!!!!".format(r=reward))
            print("Total Episode Reward: {r}".format(r=total_ep_reward))

            if done: break
