import sys, pathlib, os, datetime

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent.parent))

from urnai.trainers.filetrainer import FileTrainer

trainer = FileTrainer(os.path.join(sys.path[2], 'movetobeacon.yaml'))
trainer.pickle_black_list.append("env")
trainer.pickle_black_list.append("agent")

my_config = trainer.trainings[0]

# if trainer.trainings[0]["trainer"]["params"]["use_wandb"]:
#import wandb
#wandb.init(project="movetobeacon", name=trainer.trainings[0]["trainer"]["params"]["file_name"], entity="lpdcalves", config=my_config)

trainer.start_training()
