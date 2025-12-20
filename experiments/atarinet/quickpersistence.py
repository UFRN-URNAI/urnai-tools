import os

class QuickPersistence:
    def __init__(self, path, model, config_dict):
        self.model = model
        self.config_dict = config_dict

        if not os.path.exists(path):
            os.makedirs(path)

    def save(self, ep, eps_until_save = 200):
        if (ep + 1) % eps_until_save == 0:
            print(f"MODEL SAVED in ep: {ep}")
            self.model.save(f"saves/models/{self.config_dict['model_save_name']}/model_{ep}")

    def load(self, ep):
        self.model.load(f"saves/models/{self.config_dict['model_save_name']}/model_{ep}")