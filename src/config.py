import yaml

class Config:
    def __init__(self, path):
        with open(path, "r") as f:
            params = yaml.safe_load(f)
        for k, v in params.items():
            setattr(self, k, v)
