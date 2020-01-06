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

# TODO change to use saved umapper from other run to make sure the initialization is in the discriminative region
def evaluate_validation_performance_different_sizes(path_to_params, path_to_transformer, size_grid, n_runs_per_size):
    params = TransformParameterParser(path_to_params).parse_params()
    print(params)
    check_consistency_of_params(params)

    with open(path_to_transformer, 'rb') as f:
        data_transformer = pickle.load(f)

    metrics = init_metrics(size_grid, n_runs_per_size)
    trackers = {}
    # since I'm evaluating performance as a function of box size initialized in the discriminative region, I make umap
    # deterministic for each different run, and only vary the seed for splitting the data
    metrics_to_print = ['log_loss', 'acc', 'avg_pos_feat', 'avg_neg_feat']
    for i, size in enumerate(size_grid):
        trackers[i] = {}
        for j, run in enumerate(range(n_runs_per_size)):
            params['random_seed'] = run
            model, tracker, data_transformer = run_once_with_fixed_size(params, size, run, data_transformer)

            update_all_metrics_to_print(tracker, metrics, i , j)

            trackers[i][j] = tracker
    # add saving later
    print_all_average_metrics(metrics)
    with open(os.path.join(params['save_dir'], 'trackers_per_run.pkl'), 'wb') as f:
        pickle.dump(trackers, f)
    # just to save the grid with the metrics
    metrics['size_grid'] = size_grid
    with open(os.path.join(params['save_dir'], 'metrics_dict.pkl'), 'wb') as f:
        pickle.dump(metrics, f)

def init_metrics(size_grid, n_runs_per_size):
    # if buggy just make another smaller function
    
    metrics = {}
    metrics['log_loss'] = {}
    metrics['log_loss']['tr'] = np.zeros([size_grid.shape[0], n_runs_per_size])
    metrics['log_loss']['te'] = np.zeros([size_grid.shape[0], n_runs_per_size])

    metrics['acc'] = {}
    metrics['acc']['tr'] = np.zeros([size_grid.shape[0], n_runs_per_size])
    metrics['acc']['te'] = np.zeros([size_grid.shape[0], n_runs_per_size])

    metrics['avg_pos_feat'] = {}
    metrics['avg_pos_feat']['tr'] = np.zeros([size_grid.shape[0], n_runs_per_size])
    metrics['avg_pos_feat']['te'] = np.zeros([size_grid.shape[0], n_runs_per_size])

    metrics['avg_neg_feat'] = {}
    metrics['avg_neg_feat']['tr'] = np.zeros([size_grid.shape[0], n_runs_per_size])
    metrics['avg_neg_feat']['te'] = np.zeros([size_grid.shape[0], n_runs_per_size])
    return metrics

def update_all_metrics_to_print(tracker, metrics, size_idx, run_idx):
    splits = ['tr', 'te']
    for metric in metrics:
        for split in splits:
            metrics[metric][split][size_idx][run_idx] = tracker.metrics[split + '_' + metric][-1]


def run_once_with_fixed_size(params, size, run, data_transformer):
    start_time = time.time()

    #set_random_seeds(params) for some reason doing this produces a different UMAP embedding- likely a bug in the UMAP package I'm using, so have to set seed in data input to get consistent splits 

    if not os.path.exists(params['save_dir']):
        os.makedirs(params['save_dir'])

    with open(os.path.join(params['save_dir'], 'params.pkl'), 'wb') as f:
        pickle.dump(params, f)

    data_input = DataInput(params['data_params'])
    data_input.split_data(split_seed=params['random_seed'])

    data_input.embed_data(\
        data_transformer,
        cells_to_subsample=params['transform_params']['cells_to_subsample'], 
    ) 
    #data_input.save_transformer(params['save_dir'])
    data_input.normalize_data()

    init_gate_tree = get_init_gate_in_disc_region(size)
    model = initialize_model(params['model_params'], init_gate_tree)
    #this line fixes the size
    model.fix_size_params(size)
    data_input.convert_all_data_to_tensors()
    trackers_per_step = []
    performance_tracker = run_train_model(model, params['train_params'], data_input)
    check_size_stayed_constant(model, size)
    make_and_save_plot_to_check_umap_stays_same(model, data_input, run, params) 

    model_save_path = os.path.join(params['save_dir'], 'model.pkl')
    torch.save(model.state_dict(), model_save_path)
    
    tracker_save_path = os.path.join(params['save_dir'], 'tracker.pkl')
    with open(tracker_save_path, 'wb') as f:
        pickle.dump(performance_tracker, f)
    print('Complete main loop took %.4f seconds' %(time.time() - start_time))
    return model, performance_tracker, data_transformer


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
    gate_initializer = GateInitializer(data_input.x_tr, params['gate_init_params'])
    gate_initializer.initialize_gates() 
    gate_initializer.construct_init_gate_tree()
    gate_initializer.plot_init_gate_tree_with_data()
    plt.savefig(os.path.join(params['save_dir'], 'init_gates.png'))
    plt.clf()
    return gate_initializer.init_gate_tree

# change to match converged gate from recent good runs
def get_init_gate_in_disc_region(size):
    center = [.0493, .3147]
    return [[
        ['D1', center[0] - size/2, center[0] + size/2],
        ['D2', center[1] - size/2, center[1] + size/2]
    ]]

def check_size_stayed_constant(model, init_size):
    assert(torch.abs(torch.sigmoid(model.nodes[0].side_length_param) - init_size) <= .001)

def make_and_save_plot_to_check_umap_stays_same(model, data_input, run, params):
    results_plotter = DataAndGatesPlotterDepthOne(model, np.concatenate(data_input.x_tr))
    results_plotter.plot_data_with_gates(np.array(np.concatenate([data_input.y_tr[i] * torch.ones([data_input.x_tr[i].shape[0], 1]) for i in range(len(data_input.x_tr))])))

    plt.savefig(os.path.join(params['save_dir'], 'test%d.png' %run))
    

def print_all_average_metrics(metrics):
    for metric in metrics:
        print('Mean ' + metric.title() + ' per size for tr/te:')
        print(np.mean(metrics[metric]['tr'], axis=1), np.mean(metrics[metric]['te'], axis=1))

if __name__ == '__main__':
    path_to_params = '../configs/umap_size_eval.yaml'
    transformer_path = '../output/one_by_one_clustering_with_loss_heuristic_final_version/transformer.pkl'
    size_grid = np.array([0.025, 0.05, .1, .2, .3]) #np.array([.05, .2]) #np.array([0.025, 0.05, .1, .2, .3]) 
    n_runs_per_size = 10
    evaluate_validation_performance_different_sizes(path_to_params, transformer_path, size_grid, n_runs_per_size)
