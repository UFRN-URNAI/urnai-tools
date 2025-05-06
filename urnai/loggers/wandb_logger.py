from enum import Enum

import wandb

from urnai.loggers.logger_base import LoggerBase


class Mode(Enum):
    TRAIN = 0
    EVAL = 1

class WandbLogger(LoggerBase):
    def __init__(self):
        self.mode = Mode.TRAIN
        self.mode_name = ["train", "eval"]
        self.step_name = ["train_step", "eval_step"]
        self.num_steps = [0, 0]

        wandb.define_metric("train_step")
        wandb.define_metric("eval_step")
        wandb.define_metric("train/*", step_metric="train_step")
        wandb.define_metric("eval/*", step_metric="eval_step")

    def set_mode(self, train: bool):
        if train:
            self.mode = Mode.TRAIN
        else:
            self.mode = Mode.EVAL

    def log(self, data: dict):
        step_idx = self.mode.value
        step_name = self.step_name[step_idx]
        num_step = self.num_steps[step_idx]
        mode_name = self.mode_name[step_idx]
        wandb.log({step_name: num_step, 
                   **{f"{mode_name}/{k}": v for k,v in data.items()}})
        self.num_steps[step_idx] += 1
