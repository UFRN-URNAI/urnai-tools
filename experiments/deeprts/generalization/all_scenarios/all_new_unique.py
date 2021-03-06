import inspect
import os
import sys

from urnai.trainers.filetrainer import FileTrainer

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
parentdir = os.path.dirname(parentdir)
sys.path.insert(0, parentdir)

for filename in os.listdir(currentdir):
    if filename.endswith('json'):
        trainer = FileTrainer(currentdir + os.path.sep + filename)
        trainer.start_training()
