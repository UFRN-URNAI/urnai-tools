import inspect
import os
import sys

from urnai.trainers.filetrainer import FileTrainer

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
parentdir = os.path.dirname(parentdir)
print(parentdir)
sys.path.insert(0, parentdir)

path = \
    'D:/UFRN/Star Craft II - Reinforcement Learning/urnai-tools/urnai/solves/solve_cartpole_v1.json'
trainer = FileTrainer(path)
trainer.start_training()
