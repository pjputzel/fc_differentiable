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
from utils.DataAndGatesPlotter import MultidimDataAndGatesPlotter
from utils.DataTransformerFactory import DataTransformerFactory
from train_UMAP import run_train_model
import torch
import os
import time
from copy import deepcopy


def main_with_path(path_to_params):
    params = TransformParameterParser(path_to_params).parse_params()
    check_consistency_of_params(params)
    print(params)
    main(params)

def main(params):
    start_time = time.time()


    #evauntually uncomment this leaving asis in order ot keep the same results as before to compare.
    set_random_seeds(params)

    if not os.path.exists(params['save_dir']):
        os.makedirs(params['save_dir'])

    with open(os.path.join(params['save_dir'], 'params.pkl'), 'wb') as f:
        pickle.dump(params, f)

    data_input = DataInput(params['data_params'])
    data_input.split_data()
    print('%d samples in the training data' %len(data_input.x_tr))
    # force identity for the first transform
    data_transformer = DataTransformerFactory({'transform_type': 'identity'}, params['random_seed']).manufacture_transformer()

    data_input.embed_data_and_fit_transformer(\
        data_transformer,
        cells_to_subsample=params['transform_params']['cells_to_subsample'],
        num_cells_for_transformer=params['transform_params']['num_cells_for_transformer'],
        use_labels_to_transform_data=params['transform_params']['use_labels_to_transform_data']
    )
    # can't pickle opentsne objects
    if not params['transform_params'] == 'tsne':
        data_input.save_transformer(params['save_dir'])
    data_input.normalize_data()

    # gates aren't plotted because we're in n dimensions
    unused_cluster_gate_inits = init_gates(data_input, params)

    data_input.convert_all_data_to_tensors()
    init_gate_tree, unused_cluster_gate_inits = get_next_gate_tree(unused_cluster_gate_inits, data_input, params, model=None)
    model = initialize_model(params['model_params'], [init_gate_tree])
    trackers_per_round = []
    num_gates_left = len(unused_cluster_gate_inits)
    for i in range(num_gates_left + 1):
        performance_tracker = run_train_model(model, params['train_params'], data_input)
        trackers_per_round.append(performance_tracker.get_named_tuple_rep())
        if i == params['train_params']['num_gates_to_learn'] - 1:
            break 
        if not i == num_gates_left:
            next_gate_tree, unused_cluster_gate_inits = get_next_gate_tree(unused_cluster_gate_inits, data_input, params, model=model)
            model.add_node(next_gate_tree)

    model_save_path = os.path.join(params['save_dir'], 'model.pkl')
    torch.save(model.state_dict(), model_save_path)

    trackers_save_path = os.path.join(params['save_dir'], 'trackers.pkl')
#    trackers_per_round = [tracker.get_named_tuple_rep() for tracker in trackers_per_round]
    with open(trackers_save_path, 'wb') as f:
        pickle.dump(trackers_per_round, f)
    if params['plot_umap_reflection']:
        # reflection is about x=.5 since the data is already in umap space here
        reflected_data = []
        for data in data_input.x_tr:
            data[:, 0] = 1 - data[:, 0]
            reflected_data.append(data)
        data_input.x_tr = reflected_data
        gate_tree = model.get_gate_tree()
        reflected_gates = []
        for gate in gate_tree:
            print(gate)
            #order switches since reflected over x=.5
            low_reflected = 1 - gate[0][2]
            high_reflected = 1 - gate[0][1]
            gate[0][1] = low_reflected
            gate[0][2] = high_reflected
            print(gate)

            reflected_gates.append(gate)
        model.init_nodes(reflected_gates)
        print(model.init_nodes)
        print(model.get_gates())
    data_transformer = DataTransformerFactory(params['transform_params'], params['random_seed']).manufacture_transformer()
    data_input.convert_all_data_to_numpy()
    data_input.x_tr = data_input.x_tr_raw
    data_input.x_te = data_input.x_te_raw
    old_scale = data_input.scale
    old_offset = data_input.offset
    print("fitting projection")
    data_input.embed_data_and_fit_transformer(\
        data_transformer,
        cells_to_subsample=params['transform_params']['cells_to_subsample'],
        num_cells_for_transformer=params['transform_params']['num_cells_for_transformer'],
        use_labels_to_transform_data=params['transform_params']['use_labels_to_transform_data']
    )
    results_plotter = MultidimDataAndGatesPlotter(model, np.concatenate(data_input.x_tr), np.concatenate(data_input.untransformed_matched_x_tr), old_scale, old_offset, data_input.transformer)

    results_plotter.plot_in_feature_space(np.array(np.concatenate([data_input.y_tr[i] * torch.ones([data_input.x_tr[i].shape[0], 1]) for i in range(len(data_input.x_tr))])))
    plt.savefig(os.path.join(params['save_dir'], 'feature_results.png'))

    if params['transform_params']['embed_dim'] == 2:
        results_plotter.plot_data_with_gates(np.array(np.concatenate([data_input.y_tr[i] * torch.ones([data_input.x_tr[i].shape[0], 1]) for i in range(len(data_input.x_tr))])))
        plt.savefig(os.path.join(params['save_dir'], 'final_gates.png'))
    else:
        fig_pos, ax_pos, fig_neg, ax_neg = results_plotter.plot_data_with_gates(np.array(np.concatenate([data_input.y_tr[i] * torch.ones([data_input.x_tr[i].shape[0], 1]) for i in range(len(data_input.x_tr))])))
        with open(os.path.join(params['save_dir'], 'final_gates_pos_3d.pkl'), 'wb') as f:
            pickle.dump(fig_pos, f)

        with open(os.path.join(params['save_dir'], 'final_gates_neg_3d.pkl'), 'wb') as f:
            pickle.dump(fig_neg, f)



    with open(os.path.join(params['save_dir'], 'configs.pkl'), 'wb') as f:
        pickle.dump(params, f)

    print('Complete main loop took %.4f seconds' %(time.time() - start_time))
    return trackers_per_round[-1]


def set_random_seeds(params):
    torch.manual_seed(params['random_seed'])
    np.random.seed(params['random_seed'])


# TODO: move this function to a helper file and call it in each main!
def check_consistency_of_params(params):
    if params['train_params']['descent_type'] == 'joint_descent':
        if not params['train_params']['learning_rate_gates'] == params['train_params']['learning_rate_classifier']:
            raise ValueError('For joint descent learning rate gates and learning rate classifier must be equal')
    if params['train_params']['conv_thresh']:
        if params['train_params']['n_epoch']:
            warnings.warn('n_epoch parameter is not used when a conv_thresh is set. Training will continue until the change in loss is less than conv_thresh regardless of the number of epochs.')
    if params['transform_params']['use_labels_to_transform_data'] and not (params['transform_params']['transform_type'] == 'umap'):
        raise ValueError('Supervised data transformation only supported with umap')

def initialize_model(model_params, init_gate_tree):
    model = DepthOneModel(init_gate_tree, model_params)
    return model

def init_gates(data_input, params):
    gate_initializer = GateInitializerClustering(data_input.x_tr, params['gate_init_cluster_params'], n_dims=14)
    gate_initializer.initialize_gates() 
    gate_initializer.construct_init_gate_tree()
    print(init_gates)
    return gate_initializer.init_gate_tree

def init_plot_and_save_gates(data_input, params):
    gate_initializer = GateInitializerClustering(data_input.x_tr, params['gate_init_cluster_params'], n_dims=params['transform_params']['embed_dim'])
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
            losses.append(dummy_model(data_input.x_tr, data_input.y_tr)['loss'].cpu().detach().numpy())
        losses = np.array(losses)
        best_gate_idx = np.argmin(losses[~np.isnan(losses)])
    else:
        losses = []
        for gate_tree in unused_gate_trees:
            model = DepthOneModel([gate_tree], params['model_params'])
            performance_tracker = run_train_model(model, params['train_params'], data_input)
            losses.append(model(data_input.x_tr, data_input.y_tr)['loss'].cpu().detach().numpy())

        losses = np.array(losses)
        best_gate_idx = np.argmin(losses[~np.isnan(losses)])
    best_gate = unused_gate_trees[best_gate_idx]
    del unused_gate_trees[best_gate_idx]
    return best_gate, unused_gate_trees

def get_next_gate_tree_by_log_loss(unused_gate_trees, data_input, params, model=None):
    if model:
        losses = []
        for gate_tree in unused_gate_trees:
            dummy_model_state = deepcopy(model.state_dict())
            dummy_model = DepthOneModel(model.get_gate_tree(), params['model_params'])
            dummy_model.load_state_dict(dummy_model_state)

            dummy_model.add_node(gate_tree)
            performance_tracker = run_train_model(dummy_model, params['train_params'], data_input)
            losses.append(dummy_model(data_input.x_tr, data_input.y_tr)['log_loss'].cpu().detach().numpy())
        losses = np.array(losses)
        best_gate_idx = np.argmin(losses[~np.isnan(losses)])
    else:
        losses = []
        for gate_tree in unused_gate_trees:
            model = DepthOneModel([gate_tree], params['model_params'])
            performance_tracker = run_train_model(model, params['train_params'], data_input)
            losses.append(model(data_input.x_tr, data_input.y_tr)['log_loss'].cpu().detach().numpy())

        losses = np.array(losses)
        best_gate_idx = np.argmin(losses[~np.isnan(losses)])
    best_gate = unused_gate_trees[best_gate_idx]
    del unused_gate_trees[best_gate_idx]
    return best_gate, unused_gate_trees
        


if __name__ == '__main__':

    path_to_params = '../configs/multidim_circular.yaml'
    main_with_path(path_to_params)    

