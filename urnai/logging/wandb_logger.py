
import wandb

from urnai.logging.logger_base import LoggerBase
from urnai.logging.logging_mode_base import LoggingModeBase


class WandbLoggingMode(LoggingModeBase):
    TRAINING = "training"
    EVALUATION = "evaluation"


    @property
    def data(self):
        _data: dict = {
            WandbLoggingMode.TRAINING: {
                "metric": "train",
                "step": "train_step",
            },
            WandbLoggingMode.EVALUATION: {
                "metric": "eval",
                "step": "eval_step",
            }
        }
        return _data[self]


class WandbLogger(LoggerBase):
    mode: WandbLoggingMode
    _mode_cls: WandbLoggingMode = WandbLoggingMode
    _num_steps: dict
    
    def __init__(self, mode: str | WandbLoggingMode = None):
        if not mode:
            mode = WandbLoggingMode.TRAINING
        super().__init__(mode)
        self._num_steps = {
            WandbLoggingMode.TRAINING: 0,
            WandbLoggingMode.EVALUATION: 0,
        }

        wandb.define_metric("train_step")
        wandb.define_metric("train/*", step_metric="train_step")

        wandb.define_metric("eval_step")
        wandb.define_metric("eval/*", step_metric="eval_step")

    def log(self, data: dict) -> None:
        step_name = self.mode.data["step"]
        mode_name = self.mode.data["metric"]

        wandb.log(
            {
                step_name: self._num_steps[self.mode],
                **{
                    f"{mode_name}/{key}": value for key,value in data.items()
                }
            }
        )

        self._num_steps[self.mode] += 1
