import wandb

class QuickLog:
    def __init__(self):
        self.model_stats = {
            "total_reward" : 0,
            "actions" : {}
        }

    def log_during_ep(self, action, reward):
        self.model_stats["total_reward"] += reward
        if action in self.model_stats["actions"]:
            self.model_stats["actions"][action] += 1
        else:
            self.model_stats["actions"][action] = 1

    def end_of_ep(self):
        print("Total reward in ep: ", self.model_stats["total_reward"])
        self.model_stats["total_reward"] = 0
        self.model_stats["actions"] = {}

    def wandb_log(self, model, step, ep, wandb_logger):
        data = [[label, val] for label, val in self.model_stats["actions"].items()]
        table = wandb.Table(columns=["label", "value"], data=data)

        wandb_logger.log({
            "mean loss": model.total_loss / (step if step > 0 else 1),
            "reward per ep" : self.model_stats["total_reward"],
            "ep length" : step,
            "episode" : ep,
            "replay_buffer_len" : len(model.replay_buffer),
            "epsilon" : model.epsilon,
            "actions" : wandb.plot.bar(table, "label", "value", title="Actions"),
        })