import yaml

class TransformParameterParser:
    def __init__(self, path_to_params):
        self.path_to_params = path_to_params
    
    def parse_params(self):
        self.params = DEFAULT_PARAMS
        with open(self.path_to_params, 'rb') as f:
            new_params = yaml.safe_load(f)
        print('new params', new_params)
        for str_key in new_params:
            if type(new_params[str_key]) == dict:
                if str_key == 'transform_params':
                    self.update_transform_params(new_params[str_key])
                else:
                    self.params[str_key].update(new_params[str_key])
            else:
                self.params[str_key] = new_params[str_key]
        return self.params

    def update_transform_params(self, new_params):
        for key in new_params:
            if type(new_params[key]) == dict:
                self.params['transform_params'][key].update(new_params[key])
            else:
                self.params['transform_params'][key] = new_params[key]
                
                

DEFAULT_PARAMS =\
    {
        'random_seed': 0,
        'data_params':\
            {
                'test_percent': .2, 
            },
        'transform_params':\
            {
                'transform_type': 'umap',
                'embed_dim': 2,
                'cells_to_subsample': 10000,
                'num_cells_for_transformer': 10000000000,
                'umap_params':\
                    {
                        'n_neighbors': 15,
                        'min_dist': 0.10,
                    },
            },
        'gate_init_primkde_params':\
            {
                'n_boxes': 5,
                'prim_grid_size': 50,
                'prim_threshold_percent': .9
            },
        'gate_init_cluster_params':\
            {
                'run_kde_first': False,
                'density_threshold_percent_for_kmeans': .5,
                'n_clusters': 15,
                'kde_grid_size': 50
            },
        'gate_init_multi_heuristic_params':\
            {
                'num_gridcells_per_axis': 3,
                'num_gates': 5,
                'init_type': 'heuristic_corner',
            },
        'model_params':\
            {
                'loss_type': 'logistic',
                'node_type': 'rectangle',
                'logistic_k': 100,
                'regularisation_penalty':0. ,
                'negative_box_penalty': 0.,
                'positive_box_penalty': 0.,
                'corner_penalty': 0.,
                'init_reg_penalty': 0.,
                'feature_diff_penalty': 0.,
                'gate_size_penalty': 0.,
                'gate_size_default': (0.5, 0.5),
                'neg_proportion_default':0.0001 ,
                'classifier': True,
            },
        'plot_params':\
            {
                'marker_size': .1
            },
        'train_params':\
            {
                'metrics_to_eval': ['tr_acc', 'te_acc', 'tr_log_loss', 'te_log_loss', 'tr_avg_pos_feat', 'te_avg_pos_feat', 'tr_avg_neg_feat', 'te_avg_neg_feat'],
                'n_epoch_eval': 20,
                'n_epoch': 200,
                'learning_rate_classifier': 0.05,
                'learning_rate_gates': 0.05,
                'conv_thresh': None,
                'descent_type': 'joint_descent',
            }, 
    }
