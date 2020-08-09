import umap
import warnings
import pickle
import numpy as np
import matplotlib.pyplot as plt
from utils.TransformParameterParser import TransformParameterParser
from utils.DataInput import DataInput
from utils.MultipleGateInitializerHeuristic import MultipleGateInitializerHeuristic
from utils.DepthOneModel import DepthOneModel
from utils.DataAndGatesPlotter import DataAndGatesPlotterDepthOne
from utils.DataTransformerFactory import DataTransformerFactory
from train_UMAP import run_train_model
import torch
import os
import time

def main(path_to_params):
    start_time = time.time()

    params = TransformParameterParser(path_to_params).parse_params()
    print(params)
    check_consistency_of_params(params)

    set_random_seeds(params)

    if not os.path.exists(params['save_dir']):
        os.makedirs(params['save_dir'])

    with open(os.path.join(params['save_dir'], 'params.pkl'), 'wb') as f:
        pickle.dump(params, f)

    data_input = DataInput(params['data_params'])
    data_input.split_data()

    data_transformer = DataTransformerFactory(params['transform_params'], params['random_seed']).manufacture_transformer()
    data_input.embed_data_and_fit_transformer(\
        data_transformer,
        cells_to_subsample=params['transform_params']['cells_to_subsample'], 
    ) 
    data_input.save_transformer(params['save_dir'])
    data_input.normalize_data()
    unused_cluster_gate_inits = init_plot_and_save_gates(data_input, params)
    #everything below differs from the other main_UMAP

    init_gate_tree, unused_cluster_gate_inits = get_next_gate_tree(unused_cluster_gate_inits, model=None)
    model = initialize_model(params['model_params'], init_gate_tree)
    data_input.prepare_data_for_training()
    trackers_per_round = []
    num_gates = len(unused_cluster_gate_inits)
    for i in range(num_gates):
        performance_tracker = run_train_model(model, params['train_params'], data_input)
        if not (i == num_gates - 1):
            next_gate_tree, unused_cluster_gate_inits = get_next_gate_tree(unused_cluster_gate_inits, model=model)
            model.add_node(next_gate_tree)
        trackers_per_round.append(performance_tracker)
        
    model_save_path = os.path.join(params['save_dir'], 'model.pkl')
    torch.save(model.state_dict(), model_save_path)
    
    trackers_save_path = os.path.join(params['save_dir'], 'trackers.pkl')
    with open(tracker_save_path, 'wb') as f:
        pickle.dump(trackers_per_round, f)
    results_plotter = DataAndGatesPlotterDepthOne(model, np.concatenate(data_input.x_tr))
    #fig, axes = plt.subplots(params['gate_init_params']['n_clusters'], figsize=(1 * params['gate_init_params']['n_clusters'], 3 * params['gate_init_params']['n_clusters']))
    results_plotter.plot_data_with_gates(np.array(np.concatenate([data_input.y_tr[i] * torch.ones([data_input.x_tr[i].shape[0], 1]) for i in range(len(data_input.x_tr))])))

    plt.savefig(os.path.join(params['save_dir'], 'final_gates.png'))
    print('Complete main loop took %.4f seconds' %(time.time() - start_time))


def set_random_seeds(params):
    torch.manual_seed(params['random_seed'])
    np.random.seed(params['random_seed'])

def check_consistency_of_params(params):
    if params['train_params']['descent_type'] == 'joint_descent':
        if not params['train_params']['learning_rate_gates'] == params['train_params']['learning_rate_classifier']:
            raise ValueError('For joint descent learning rate gates and learning rate classifier must be equal')
    if params['train_params']['conv_thresh']:
        if params['train_params']['n_epoch']:
            warnings.warn('n_epoch parameter is not used when a conv_thresh is set. Training will continue until the change in loss is less than conv_thresh regardless of the number of epochs.')

def initialize_model(model_params, init_gate_tree):
    model = DepthOneModel(init_gate_tree, model_params)
    return model

def init_plot_and_save_gates(data_input, params):
    gate_initializer = GateInitializerPrimKDE(data_input.x_tr, params['gate_init_params'])
    gate_initializer.initialize_gates() 
    gate_initializer.construct_init_gate_tree()
    gate_initializer.plot_init_gate_tree_with_data()
    plt.savefig(os.path.join(params['save_dir'], 'init_gates.png'))
    plt.clf()
    return gate_initializer.init_gate_tree

def get_next_gate_tree(unused_gate_trees, params, model=None):
    if model:
        losses = []
        for gate_tree in unused_gate_trees:
            dummy_model = deepcopy(model)
            dummy_model.add_node(gate_tree)
            fit_classifier_params(dummy_model, data_input, params['train_params']['learning_rate_classifier'])
            losses.append(dummy_model(data_input.x_tr, data_input.y_tr)['log_loss'].cpu().detach().numpy())
        best_gate_idx = np.argmax(np.array(losses))
    else:
        losses = []
        for gate_tree in unused_gate_trees:
            model = DepthOneModel([gate_tree], params['model_params'])
            fit_classifier_params(model, data_input, params['train_params']['learning_rate_classifier'])
            losses.append(model(data_input.x_tr, data_input.y_tr)['log_loss'].cpu().detach().numpy())
        best_gate_idx = np.argmax(np.array(losses))
    best_gate = unused_gate_trees[best_gate_idx]
    del unused_gate_trees[best_gate_idx]
    return best_gate, unused_gate_trees
        


if __name__ == '__main__':
    path_to_params = '../configs/umap_clustering_default.yaml'
    main(path_to_params)    


