import copy
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
import matplotlib
 


def normalize_data(data):
    x_min = data.min(axis=0)
    x_max = data.max(axis=0)
    offset = x_min
    scale = x_max - x_min
    normalized_data = (data - offset)/scale
    return normalized_data

def plot_data_inside_semi_synth_gate_in_real_UMAP_space(path_to_semi_synth_params, path_to_params_real):

    params_semi_synth, model_semi_synth, data_input_semi_synth, umapper_semi_synth = load_saved_model_and_matching_data_input(path_to_semi_synth_params)


    params_real, model_real, data_input_real, umapper_real = load_saved_model_and_matching_data_input(path_to_params_real)

    plotter_semi_synth = DataAndGatesPlotterDepthOne(model_semi_synth, np.concatenate(data_input_semi_synth.x_tr))
    semi_synth_data_inside_gate = get_data_inside_gate(plotter_semi_synth, data_input_semi_synth, model_semi_synth)

    data_inside_semi_synth_in_real_umapper_space = umapper_real.transform(semi_synth_data_inside_gate)
    data_inside_semi_synth_in_real_umapper_space = normalize_data(data_inside_semi_synth_in_real_umapper_space)
    #labels_in_real_umapper_space = data_input.y_tr[data_inside_semi_synth_in_real_umapper_space_idxs]    


    plotter_real_data = DataAndGatesPlotterDepthOne(model_real, np.concatenate(data_inside_semi_synth_in_real_umapper_space))
    plotter_real_data.plot_all_gates(plt.gca())
    size = 1000 * 1/data_inside_semi_synth_in_real_umapper_space.shape[0]
    plt.scatter(data_inside_semi_synth_in_real_umapper_space[:, 0], data_inside_semi_synth_in_real_umapper_space[:, 1], s=size)
    plt.savefig('semi_synth_inside_gate_in_real_umapper_space.png')
    #plotter_real_data.plot_data_with_gates(labels_in_real_umapper_space)


def get_data_inside_both_visually_correct_and_learned_gate(path_to_params_learned):
    matplotlib.rcParams.update({'font.size': 22})


    path_to_params_learned, model_learned, data_input = load_saved_model_and_matching_data_input(path_to_params_learned)

    vis_correct_gate = [ [['D1', 0. , model_learned.get_gate_tree()[0][0][2] ], ['D2', model_learned.get_gate_tree()[0][1][1] , .75]] ] 
    vis_correct_model = DepthOneModel(vis_correct_gate, path_to_params_learned['model_params'])

    plotter_model = DataAndGatesPlotterDepthOne(model_learned, np.concatenate(data_input.x_tr))
    plotter_vis_corr = DataAndGatesPlotterDepthOne(vis_correct_model, np.concatenate(data_input.x_tr))

    model_learned_data_inside_gate = get_data_inside_gate(plotter_model, data_input, model_learned)
    vis_corr_data_inside_gate = get_data_inside_gate(plotter_vis_corr, data_input, vis_correct_model)

    with open('model_feat_diff_data_inside_gate.pkl', 'wb') as f:
        pickle.dump(model_learned_data_inside_gate, f)


    with open('vis_corr_data_inside_gate.pkl', 'wb') as f:
        pickle.dump(vis_corr_data_inside_gate, f)
    
    return model_learned_data_inside_gate, vis_corr_data_inside_gate

# returns data inside the original feature space inside the learned gate in umap space
def get_data_inside_gate(plotter, data_input, model, ret_bool_idxs=False):
    # plotter data is the concatenated training data from data_input
    data_inside_first_gate_idxs = plotter.filter_data_at_single_node(plotter.data, plotter.model.nodes[0], return_idxs=True)        
    if ret_bool_idxs:
        return data_inside_first_gate_idxs

    untransformed_x_tr = np.concatenate(data_input.untransformed_matched_x_tr)
    data_inside_first_gate_inverse_transform = untransformed_x_tr[data_inside_first_gate_idxs]
    return data_inside_first_gate_inverse_transform

def load_saved_model_and_matching_data_input(path_to_params):
    def set_random_seeds(params):
        torch.manual_seed(params['random_seed'])
        np.random.seed(params['random_seed'])
    start_time = time.time()

    params = TransformParameterParser(path_to_params).parse_params()
    print(params)

    #evauntually uncomment this leaving asis in order ot keep the same results as before to compare.
    set_random_seeds(params)

    data_input = DataInput(params['data_params'])
    data_input.split_data()
    print('%d samples in the training data' %len(data_input.x_tr))

    with open(os.path.join(params['save_dir'], 'trackers.pkl'), 'rb') as f:
        trackers = pickle.load(f)

    with open(os.path.join(params['save_dir'], 'transformer.pkl'), 'rb') as f:
        umapper = pickle.load(f)
    # FOR DEBUGGING ONLY
    #params['transform_params']['cells_to_subsample'] = 10
    data_input.embed_data(\
        umapper,
        cells_to_subsample=params['transform_params']['cells_to_subsample'],
        use_labels_to_transform_data=params['transform_params']['use_labels_to_transform_data']
    ) 
    data_input.normalize_data()
    data_input.convert_all_data_to_tensors()

    model = DepthOneModel([[['D1', 0, 0], ['D2', 0, 0]]], params['model_params'])
    model.load_state_dict(torch.load(os.path.join(params['save_dir'], 'model.pkl')))
    return params, model, data_input, umapper


if __name__ == '__main__':
    ### to get data inside the 'vis correct' and real data
    #path_to_params = '../configs/umap_with_feat_diff_reg.yaml'
    #get_data_inside_both_visually_correct_and_learned_gate(path_to_params)

    ### to get plot of data inside semi synth learned gate in real data umap space
    path_to_params_semi = '../configs/umap_semi_synth.yaml'
    path_to_params_real = '../configs/umap_with_feat_diff_reg.yaml'
    plot_data_inside_semi_synth_gate_in_real_UMAP_space(path_to_params_semi, path_to_params_real)
    
