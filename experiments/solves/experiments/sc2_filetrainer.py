import sys, pathlib, os, datetime
import wandb
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent.parent))

from urnai.trainers.filetrainer import FileTrainer

trainer = FileTrainer(os.path.join(sys.path[2], 'sc2_move_to_beacon.yaml'))

my_config = trainer.trainings[0]

wandb.init(project="move-to-beacon", name=str(datetime.datetime.now()), entity="lpdcalves", config=my_config)

trainer.start_training()
