import umap
import warnings
import pickle
import numpy as np
import matplotlib.pyplot as plt
from train_UMAP import fit_classifier_params
from utils.TransformParameterParser import TransformParameterParser
from utils.DataInput import DataInput
from utils.GateInitializerPrimKDE import GateInitializerPrimKDE
from utils.GateInitializerClustering import GateInitializerClustering
from utils.DepthOneModel import DepthOneModel
from utils.DataAndGatesPlotter import DataAndGatesPlotterDepthOne
from utils.DataTransformerFactory import DataTransformerFactory
from train_UMAP import run_train_model
import torch
import os
import time
from copy import deepcopy

def main(path_to_params):
    start_time = time.time()

    params = TransformParameterParser(path_to_params).parse_params()
    print(params)
    check_consistency_of_params(params)

    #evauntually uncomment this leaving asis in order ot keep the same results as before to compare.
    #set_random_seeds(params)

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
        num_cells_for_transformer=params['transform_params']['num_cells_for_transformer'] 
    ) 
    data_input.save_transformer(params['save_dir'])
    data_input.normalize_data()
    unused_cluster_gate_inits = init_plot_and_save_gates(data_input, params)
    #everything below differs from the other main_UMAP
    data_input.convert_all_data_to_tensors()
    init_gate_tree, unused_cluster_gate_inits = get_next_gate_tree(unused_cluster_gate_inits, data_input, params, model=None)
    model = initialize_model(params['model_params'], [init_gate_tree])
    trackers_per_round = []
    num_gates_left = len(unused_cluster_gate_inits)
    #print(num_gates_left, 'asdfasdfasdfasdfasdfasdfas')
    for i in range(num_gates_left + 1):
        performance_tracker = run_train_model(model, params['train_params'], data_input)
        if not i == num_gates_left:
            next_gate_tree, unused_cluster_gate_inits = get_next_gate_tree(unused_cluster_gate_inits, data_input, params, model=model)
            model.add_node(next_gate_tree)
        trackers_per_round.append(performance_tracker)
        
    model_save_path = os.path.join(params['save_dir'], 'model.pkl')
    torch.save(model.state_dict(), model_save_path)
    
    trackers_save_path = os.path.join(params['save_dir'], 'trackers.pkl')
    with open(trackers_save_path, 'wb') as f:
        pickle.dump(trackers_per_round, f)
    results_plotter = DataAndGatesPlotterDepthOne(model, np.concatenate(data_input.x_tr))
    #fig, axes = plt.subplots(params['gate_init_params']['n_clusters'], figsize=(1 * params['gate_init_params']['n_clusters'], 3 * params['gate_init_params']['n_clusters']))
    results_plotter.plot_data_with_gates(np.array(np.concatenate([data_input.y_tr[i] * torch.ones([data_input.x_tr[i].shape[0], 1]) for i in range(len(data_input.x_tr))])))

    plt.savefig(os.path.join(params['save_dir'], 'final_gates.png'))


    with open(os.path.join(params['save_dir'], 'configs.pkl'), 'wb') as f:
        pickle.dump(params, f)

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
    gate_initializer = GateInitializerClustering(data_input.x_tr, params['gate_init_cluster_params'])
    gate_initializer.initialize_gates() 
    gate_initializer.construct_init_gate_tree()
    gate_initializer.plot_init_gate_tree_with_data()
    plt.savefig(os.path.join(params['save_dir'], 'init_gates.png'))
    plt.clf()
    return gate_initializer.init_gate_tree

def get_next_gate_tree(unused_gate_trees, data_input, params, model=None):
    if model:
        losses = []
        for gate_tree in unused_gate_trees:
            dummy_model_state = deepcopy(model.state_dict())
            dummy_model = DepthOneModel(model.get_gate_tree(), params['model_params'])
            dummy_model.load_state_dict(dummy_model_state)

            dummy_model.add_node(gate_tree)
            performance_tracker = run_train_model(dummy_model, params['train_params'], data_input)
            losses.append(dummy_model(data_input.x_tr, data_input.y_tr)['log_loss'].cpu().detach().numpy())
        best_gate_idx = np.argmin(np.array(losses))
    else:
        losses = []
        for gate_tree in unused_gate_trees:
            model = DepthOneModel([gate_tree], params['model_params'])
            performance_tracker = run_train_model(model, params['train_params'], data_input)
            losses.append(model(data_input.x_tr, data_input.y_tr)['log_loss'].cpu().detach().numpy())
        best_gate_idx = np.argmin(np.array(losses))
    best_gate = unused_gate_trees[best_gate_idx]
    del unused_gate_trees[best_gate_idx]
    return best_gate, unused_gate_trees
        


if __name__ == '__main__':
    path_to_params = '../configs/umap_clustering_default.yaml'
    main(path_to_params)    

