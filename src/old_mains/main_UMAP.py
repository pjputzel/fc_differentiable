import umap
import pickle
import numpy as np
import matplotlib.pyplot as plt
from utils.TransformParameterParser import TransformParameterParser
from utils.DataInput import DataInput
from utils.GateInitializer import GateInitializer
from utils.DepthOneModel import DepthOneModel
from utils.DataAndGatesPlotter import DataAndGatesPlotterDepthOne
from utils.DataTransformerFactory import DataTransformerFactory
from train_UMAP import run_train_model
import torch
import os
import time

def main(path_to_params):
    params = TransformParameterParser(path_to_params).parse_params()
    start_time = time.time()
    print(params)
    if not os.path.exists(params['save_dir']):
        os.makedirs(params['save_dir'])
    with open(os.path.join(params['save_dir'], 'params.pkl'), 'wb') as f:
        pickle.dump(params, f)
    data_input = DataInput(params['data_params'])
    data_input.split_data()

    data_transformer = DataTransformerFactory(params['transform_params']).manufacture_transformer()
    data_input.embed_data(data_transformer, params['transform_params']['cells_to_subsample'], params['transform_params']['num_cells_for_transformer']) #cells to subsample should change to a transformer param instead
    data_input.save_transformer(params['save_dir'])

    data_input.normalize_data()
    init_gate_tree = init_plot_and_save_gates(data_input, params)

    model = initialize_model(params['model_params'], init_gate_tree)
    data_input.prepare_data_for_training()
    performance_tracker = run_train_model(model, params['train_params'], data_input)
    
    model_save_path = os.path.join(params['save_dir'], 'model.pkl')
    torch.save(model.state_dict(), model_save_path)
    
    tracker_save_path = os.path.join(params['save_dir'], 'tracker.pkl')
    with open(tracker_save_path, 'wb') as f:
        pickle.dump(performance_tracker, f)
    results_plotter = DataAndGatesPlotterDepthOne(model, np.concatenate(data_input.x_tr))
    #fig, axes = plt.subplots(params['gate_init_params']['n_clusters'], figsize=(1 * params['gate_init_params']['n_clusters'], 3 * params['gate_init_params']['n_clusters']))
    results_plotter.plot_data_with_gates(np.array(np.concatenate([data_input.y_tr[i] * torch.ones([data_input.x_tr[i].shape[0], 1]) for i in range(len(data_input.x_tr))])))

    plt.savefig(os.path.join(params['save_dir'], 'final_gates.png'))
    print('Complete main loop took %.4f seconds' %(time.time() - start_time))

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

if __name__ == '__main__':
    path_to_params = '../configs/umap_default.yaml'
    main(path_to_params)    


    
    
