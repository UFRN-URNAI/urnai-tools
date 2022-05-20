import sys, pathlib, os
import wandb
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent.parent))

from urnai.trainers.filetrainer import FileTrainer

trainer = FileTrainer(os.path.join(sys.path[2], 'sc2_defeat_roaches_simple.yaml'))

my_config = trainer.trainings[0]

wandb.init(project="action-branching", entity="lpdcalves", config=my_config)

trainer.start_training()
