import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 

from agents.actions.sc2_wrapper import SC2Wrapper, TerranWrapper, ProtossWrapper
from agents.states.sc2 import Simple64State

from models.model_builder import ModelBuilder

from utils.logger import Logger

state_builder = Simple64State()
action_wrapper = TerranWrapper()

helper = ModelBuilder()  
helper.add_input_layer(int(state_builder.get_state_dim()))
helper.add_fullyconn_layer(512)
helper.add_fullyconn_layer(256)
helper.add_output_layer(action_wrapper.get_action_space_dim())

b = Logger(10000)
print("Saving b")
b.save(".")
print(b.ep_total)

b.ep_total = 900
print(b.ep_total)

print("Loading b")
b.load(".")
print(b.ep_total)


print(a.build_model)
print(a.model_layers)
print("Saving a")
a.save(".")
 
print("Loading a")
a.load(".")
print(a.build_model)
print(a.model_layers)
