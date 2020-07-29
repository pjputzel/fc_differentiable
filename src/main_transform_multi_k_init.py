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
    set_random_seeds(params)

    if not os.path.exists(params['save_dir']):
        os.makedirs(params['save_dir'])

    with open(os.path.join(params['save_dir'], 'params.pkl'), 'wb') as f:
        pickle.dump(params, f)

    data_input = DataInput(params['data_params'])
    data_input.split_data()
    print('%d samples in the training data' %len(data_input.x_tr))
    data_transformer = DataTransformerFactory(params['transform_params'], params['random_seed']).manufacture_transformer()

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

    potential_gates = get_all_potential_gates(data_input, params)
    data_input.convert_all_data_to_tensors()
    model, remaining_gates, tracker = initialize_model_with_best_gate(
        potential_gates, data_input, params
    )
    trackers_per_round = [tracker]
    for i in range(params['train_params']['num_gates_to_learn'] - 1):
        best_gate, remaining_gates, tracker = get_next_best_gate(
            remaining_gates, data_input, params, model
        )
        model.add_node(best_gate)
        trackers_per_round.append(tracker.get_named_tuple_rep())


 #   if params['transform_params']['embed_dim'] == 3:
 #       unused_cluster_gate_inits = init_gates(data_input, params)
 #   else:
 #       unused_cluster_gate_inits = init_plot_and_save_gates(data_input, params)
 #   #everything below differs from the other main_UMAP
 #   data_input.convert_all_data_to_tensors()
 #   init_gate_tree, unused_cluster_gate_inits = get_next_gate_tree(unused_cluster_gate_inits, data_input, params, model=None)
 #   model = initialize_model(params['model_params'], [init_gate_tree])
 #   trackers_per_round = []
 #   num_gates_left = len(unused_cluster_gate_inits)
 #   #print(num_gates_left, 'asdfasdfasdfasdfasdfasdfas')
 #   for i in range(num_gates_left + 1):
 #       performance_tracker = run_train_model(model, params['train_params'], data_input)
 #       trackers_per_round.append(performance_tracker.get_named_tuple_rep())
 #       if i == params['train_params']['num_gates_to_learn'] - 1:
 #           break 
 #       if not i == num_gates_left:
 #           next_gate_tree, unused_cluster_gate_inits = get_next_gate_tree(unused_cluster_gate_inits, data_input, params, model=model)
 #           model.add_node(next_gate_tree)
        
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
    results_plotter = DataAndGatesPlotterDepthOne(model, np.concatenate(data_input.x_tr))
    #fig, axes = plt.subplots(params['gate_init_params']['n_clusters'], figsize=(1 * params['gate_init_params']['n_clusters'], 3 * params['gate_init_params']['n_clusters']))



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

def initialize_model_with_best_gate(potential_gates, data_input, params):
    losses = []
    models = []
    for g, gate in enumerate(potential_gates):
        model = DepthOneModel([gate], params['model_params'])
        tracker = run_train_model(
            model, params['train_params'], data_input
        )
        losses.append(
            model(
                data_input.x_tr, data_input.y_tr
            )['loss'].cpu().detach().numpy()
        )
        models.append(model)
    best_gate_idx = np.argmin(np.array(losses)[~np.isnan(losses)])
    best_model = models[best_gate_idx]
    remaining_gates = [
        gate for g, gate in enumerate(potential_gates)
        if not g == best_gate_idx
    ]

    return best_model, remaining_gates, tracker

def get_next_best_gate(remaining_gates, data_input, params, model):
    losses = []
    trackers = []
    for gate in remaining_gates:
        dummy_model_state = deepcopy(model.state_dict())
        init_gates = rectangularize_gates(model)
        dummy_model = DepthOneModel(init_gates, params['model_params'])
        dummy_model.load_state_dict(dummy_model_state)
        dummy_model.add_node(gate)
        trackers.append(run_train_model(
            dummy_model, params['train_params'], data_input
        ))
        losses.append(
            dummy_model(
                data_input.x_tr, data_input.y_tr
            )['loss'].cpu().detach().numpy()
        )
    losses = np.array(losses)
    best_gate_idx = np.argmin(losses[~np.isnan(losses)])

    best_gate = remaining_gates[best_gate_idx]
    remaining_gates = [
        gate for g, gate in enumerate(remaining_gates)
        if not g == best_gate_idx
    ]
    best_tracker = trackers[best_gate_idx]
    return best_gate, remaining_gates, best_tracker

def rectangularize_gates(model):
    gates = model.get_gate_tree()
    node_type = model.node_type
    rectangular_gates = []
    for gate in gates:
        if node_type == 'circular':
            centerx = gate[0][0]
            centery = gate[0][1]
            r = gate[1]
            rect_gate = [
                ['D1', centerx - r, centerx + r],
                ['D2', centery - r, centery + r]
            ]
                        
        elif node_type == 'elliptical':
            raise NotImplementedError('only circle and square gates implemented for multi-gates currently')
        elif node_type == 'axis_aligned_elliptical':
            raise NotImplementedError('only circle and square gates implemented for multi-gates currently')
        elif (node_type == 'rectangle') or (node_type == 'square'):
            pass
        else:
            raise ValueError('Gate type %s not recognize' %node_type)
        rectangular_gates.append(rect_gate)

    return rectangular_gates

    

#def initialize_model(model_params, init_gate_tree):
#    model = DepthOneModel(init_gate_tree, model_params)
#    return model
#
#def init_gates(data_input, params):
#    gate_initializer = GateInitializerClustering(data_input.x_tr, params['gate_init_cluster_params'], n_dims=params['transform_params']['embed_dim'])
#    gate_initializer.initialize_gates() 
#    gate_initializer.construct_init_gate_tree()
#
#    return gate_initializer.init_gate_tree
#
#def init_plot_and_save_gates(data_input, params):
#    gate_initializer = GateInitializerClustering(data_input.x_tr, params['gate_init_cluster_params'], n_dims=params['transform_params']['embed_dim'])
#    gate_initializer.initialize_gates() 
#    gate_initializer.construct_init_gate_tree()
#    gate_initializer.plot_init_gate_tree_with_data()
#    plt.savefig(os.path.join(params['save_dir'], 'init_gates.png'))
#    plt.clf()
#
#    return gate_initializer.init_gate_tree


def get_all_potential_gates(data_input, params):
    potential_gates = []
    for k in params['gate_init_cluster_params']['multi_k_init_values']:
        params_k = params['gate_init_cluster_params']
        params_k['n_clusters'] = k
        gate_initializer = GateInitializerClustering(
            data_input.x_tr, params_k, 
            n_dims=params['transform_params']['embed_dim']
        )
        gate_initializer.initialize_gates() 
        gate_initializer.construct_init_gate_tree()
        potential_gates.extend(gate_initializer.init_gate_tree)
    return potential_gates

#def get_next_gate_tree(unused_gate_trees, data_input, params, model=None):
#    if model:
#        losses = []
#        for gate_tree in unused_gate_trees:
#            dummy_model_state = deepcopy(model.state_dict())
#            dummy_model = DepthOneModel(model.get_gate_tree(), params['model_params'])
#            dummy_model.load_state_dict(dummy_model_state)
#
#            dummy_model.add_node(gate_tree)
#            performance_tracker = run_train_model(dummy_model, params['train_params'], data_input)
#            losses.append(dummy_model(data_input.x_tr, data_input.y_tr)['loss'].cpu().detach().numpy())
#        losses = np.array(losses)
#        best_gate_idx = np.argmin(losses[~np.isnan(losses)])
#    else:
#        losses = []
#        for gate_tree in unused_gate_trees:
#            model = DepthOneModel([gate_tree], params['model_params'])
#            performance_tracker = run_train_model(model, params['train_params'], data_input)
#            losses.append(model(data_input.x_tr, data_input.y_tr)['loss'].cpu().detach().numpy())
#
#        losses = np.array(losses)
#        best_gate_idx = np.argmin(losses[~np.isnan(losses)])
#    best_gate = unused_gate_trees[best_gate_idx]
#    del unused_gate_trees[best_gate_idx]
#    return best_gate, unused_gate_trees
#
#def get_next_gate_tree_by_log_loss(unused_gate_trees, data_input, params, model=None):
#    if model:
#        losses = []
#        for gate_tree in unused_gate_trees:
#            dummy_model_state = deepcopy(model.state_dict())
#            dummy_model = DepthOneModel(model.get_gate_tree(), params['model_params'])
#            dummy_model.load_state_dict(dummy_model_state)
#
#            dummy_model.add_node(gate_tree)
#            performance_tracker = run_train_model(dummy_model, params['train_params'], data_input)
#            losses.append(dummy_model(data_input.x_tr, data_input.y_tr)['log_loss'].cpu().detach().numpy())
#        losses = np.array(losses)
#        best_gate_idx = np.argmin(losses[~np.isnan(losses)])
#    else:
#        losses = []
#        for gate_tree in unused_gate_trees:
#            model = DepthOneModel([gate_tree], params['model_params'])
#            performance_tracker = run_train_model(model, params['train_params'], data_input)
#            losses.append(model(data_input.x_tr, data_input.y_tr)['log_loss'].cpu().detach().numpy())
#
#        losses = np.array(losses)
#        best_gate_idx = np.argmin(losses[~np.isnan(losses)])
#    best_gate = unused_gate_trees[best_gate_idx]
#    del unused_gate_trees[best_gate_idx]
#    return best_gate, unused_gate_trees
        


if __name__ == '__main__':

#    path_to_params = '../configs/umap_with_feat_diff_reg.yaml'

#    path_to_params = '../configs/umap_semi_synth.yaml'
#    path_to_params = '../configs/umap_circular.yaml'
#    path_to_params = '../configs/umap_tsne.yaml'
#    path_to_params = '../configs/umap_3d.yaml'
#    path_to_params = '../configs/umap_pca.yaml'
#    path_to_params = '../configs/umap_axis_aligned_elliptical.yaml'

#    path_to_params = '../configs/umap_elliptical.yaml'
#    path_to_params = '../configs/umap_with_presplit.yaml'

    path_to_params = '../configs/umap_BALL.yaml'
    main(path_to_params)    

