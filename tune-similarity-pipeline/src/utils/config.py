# config.py

import yaml

class Config:
    def __init__(self, config_file):
        self.config_file = config_file
        self.settings = self.load_config()

    def load_config(self):
        with open(self.config_file, 'r') as file:
            return yaml.safe_load(file)

    def get(self, key, default=None):
        return self.settings.get(key, default)