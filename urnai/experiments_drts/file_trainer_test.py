import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
parentdir = os.path.dirname(parentdir)
print(parentdir)
sys.path.insert(0,parentdir)

# import tensorflow as tf
# physical_devices = tf.config.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(physical_devices[0], True)

from urnai.trainers.filetrainer import FileTrainer

#trainer = FileTrainer("D:/UFRN/Star Craft II - Reinforcement Learning/urnai-tools/urnai/solves/experiments/solve_simple64_test.csv")
trainer = FileTrainer("D:/UFRN/Star Craft II - Reinforcement Learning/urnai-tools/urnai/solves/solve_cartpole_v1.json")
trainer.start_training()