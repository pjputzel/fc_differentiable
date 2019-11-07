import yaml

DEFAULT_PARAMS = {
    'n_neighbors': 15,
    'min_dist': .1
    }
class UmapParameterParser:
    def __init__(self, path_to_params):
        self.path_to_params = path_to_params

    def parse_params(self):
        params = DEFAULT_PARAMS
        with open(path_to_params, 'rb') as f:
            new_params = yaml.safe_load(f)
        params.update(new_params)
        self.params = params
        return params
