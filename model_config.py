import json


class ModelParams:

    def __init__(self, ):
        self.test_filenames = None
        self.interval_val = None
        self.enable_mc_dropout = None
        self.model_type = None
        self.targets = None
        self.features = None
        self.maskval = None
        self.n_features = None
        self.timesteps = None
        self.npb = None
        self.curve_range = None

    def load(self, params_path="model_params.json"):
        f = open(params_path, "r")
        model_params = json.load(f)
        self.npb = model_params["npb"]
        self.timesteps = model_params["timesteps"]
        self.n_features = model_params["n_features"]
        self.maskval = model_params["maskval"]
        self.features = model_params["features"]
        self.targets = model_params["targets"]
        self.curve_range = model_params["curve_range"]
        self.model_type = model_params["model_type"]
        self.enable_mc_dropout = model_params["enable_mc_droput"]
        self.interval_val = model_params["interval_val"]
        self.test_filenames = model_params["test_filenames"]
        f.close()


params = ModelParams()

