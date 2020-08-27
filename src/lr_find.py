import umap
from torch.utils.data import DataLoader
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
from torch_lr_finder import LRFinder


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

    data_input.normalize_data()

    # gates aren't plotted because we're in n dimensions
    unused_cluster_gate_inits = init_gates(data_input, params)

    # data_input.convert_all_data_to_tensors()
    figscale = 8
    fig, axs = plt.subplots(nrows=len(unused_cluster_gate_inits), figsize=(figscale, len(unused_cluster_gate_inits)*figscale))

    print("initializing model")
    for gate, ax in zip(unused_cluster_gate_inits, axs):
        dataset = torch.utils.data.TensorDataset(torch.tensor(data_input.x_tr, dtype=torch.float),
                                                 torch.tensor(data_input.y_tr, dtype=torch.float))
        trainloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
        criterion = torch.nn.BCEWithLogitsLoss()
        model = SingleGateModel(params, gate)

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-7, weight_decay=1e-2)

        print("initializing LR finder")
        lr_finder = LRFinder(model, optimizer, criterion)
        lr_finder.range_test(trainloader, end_lr=1e4, num_iter=100)
        lr_finder.plot(ax=ax)
        print("LR History:", lr_finder.history)
    plt.savefig(os.path.join(params['save_dir'], 'lr_find.png'))

    print('Complete main loop took %.4f seconds' %(time.time() - start_time))
    return

class AllGatesModel(torch.nn.Module):
    def __init__(self, params, gates):
        super(AllGatesModel, self).__init__()
        self.gates = gates
        self.models = torch.nn.ModuleList([initialize_model(params['model_params'], [init_gate_tree]) for init_gate_tree in self.gates])

    def forward(self, x):
        output = self.models[0](x)
        pred = torch.zeros_like(output['y_pred'])
        for model in self.models:
            output = model(x)
            pred += output['y_pred']

        return pred/len(self.models)

class SingleGateModel(torch.nn.Module):
    def __init__(self, params, gate):
        super(SingleGateModel, self).__init__()
        self.gate = gate
        self.model = initialize_model(params['model_params'], [gate])

    def forward(self, x):
        output = self.model(x)

        return output['y_pred']

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

    path_to_params = '../configs/multidim_rect.yaml'
    main_with_path(path_to_params)    

