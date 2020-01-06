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





def test_plot(path_to_params):
    params = TransformParameterParser(path_to_params).parse_params()
    params['data_params']['path_to_x_list'] = '../data/cll/x_UMAP_dev_not_swapped.pkl'
    #params['transform_params']['cells_to_subsample'] = 100 #UNCOMMENT ME JUST FOR DEBUGGING
    data_input = DataInput(params['data_params'])
    data_input.split_data()

    path_to_model = os.path.join(params['save_dir'], 'model.pkl')
    path_to_umapper = os.path.join(params['save_dir'], 'transformer.pkl')
    model = load_saved_model(path_to_model, params['model_params'])
    umapper = load_umapper(path_to_umapper)

    data_input.embed_data(\
        umapper,
        cells_to_subsample=params['transform_params']['cells_to_subsample'],
    )
    data_input.normalize_data()
    catted_data = np.concatenate(data_input.x_tr)
    plotter = DataAndGatesPlotterDepthOne(model, catted_data)
    gate_data_idxs = [[2, 3], [0, 1], [5, 6], [10, 7]]
    plotter.plot_inverse_UMAP_transform_in_feature_space_with_filtering(umapper, np.concatenate(data_input.untransformed_matched_x_tr), gate_data_idxs=gate_data_idxs, ms=5)
    plt.savefig('points_in_original_feature_space_with_filtering.png')
    plotter.plot_inverse_UMAP_transform_in_feature_space(umapper, np.concatenate(data_input.untransformed_matched_x_tr), gate_data_idxs=gate_data_idxs, ms=.01)
    plt.savefig('points_in_original_feature_space.png')



def load_umapper(path_to_umapper):
    with open(path_to_umapper, 'rb') as f:
        umapper = pickle.load(f)
    return umapper

def load_saved_model(path_to_model, model_params):
    with open(path_to_model, 'rb') as f:
        state_dict = torch.load(f)
    model = DepthOneModel(GateInitializerClustering.get_fake_init_gates(5), model_params)
    model.load_state_dict(state_dict)
    return model

if __name__ == '__main__':
    path_to_params = '../configs/umap_clustering_default.yaml'
    test_plot(path_to_params)
