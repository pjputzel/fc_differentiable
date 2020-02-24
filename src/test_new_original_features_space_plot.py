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
    #params['data_params']['path_to_x_list'] = '../data/cll/x_UMAP_dev_not_swapped.pkl'
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
    catted_data_pos = np.concatenate([data_input.x_tr[i] for i in range(len(data_input.x_tr)) if data_input.y_tr[i] == 1])
    catted_data_neg = np.concatenate([data_input.x_tr[i] for i in range(len(data_input.x_tr)) if data_input.y_tr[i] == 0])
    catted_data = np.concatenate(data_input.x_tr)


    plotter = DataAndGatesPlotterDepthOne(model, catted_data)
    gate_data_idxs = [[3, 4], [0, 2], [6, 7], [11, 8], [0, 1], [2, 3], [-1, 5]]
    #plotter.plot_inverse_UMAP_transform_in_feature_space_with_filtering(umapper, np.concatenate(data_input.untransformed_matched_x_tr), gate_data_idxs=gate_data_idxs, ms=5)
    #plt.savefig('points_in_original_feature_space_with_filtering.png')
    #plt.clf()
    plotter.plot_inverse_UMAP_transform_in_feature_space(umapper, np.concatenate(data_input.untransformed_matched_x_tr), gate_data_idxs=gate_data_idxs, ms=.01)
    plt.savefig('points_in_original_feature_space.png')
    plt.clf()

    plotter_pos = DataAndGatesPlotterDepthOne(model, catted_data_pos)
    plotter_pos.plot_inverse_UMAP_transform_in_feature_space(umapper, np.concatenate([data_input.untransformed_matched_x_tr[i] for i in range(len(data_input.x_tr)) if data_input.y_tr[i] == 1]), gate_data_idxs=gate_data_idxs, ms=5)
    plt.savefig('points_in_original_feature_space_pos.png')
    plt.clf()
    
    plotter_neg = DataAndGatesPlotterDepthOne(model, catted_data_neg)
    plotter_neg.plot_inverse_UMAP_transform_in_feature_space(umapper, np.concatenate([data_input.untransformed_matched_x_tr[i] for i in range(len(data_input.x_tr)) if data_input.y_tr[i] == 0]), gate_data_idxs=gate_data_idxs, ms=5)
    plt.savefig('points_in_original_feature_space_neg.png')

    plotter_extra_sample1 = DataAndGatesPlotterDepthOne(model, data_input.x_tr[0])
    plotter_extra_sample1.plot_inverse_UMAP_transform_in_feature_space(umapper, data_input.untransformed_matched_x_tr[0], gate_data_idxs=gate_data_idxs, ms=5)
    plt.savefig('points_in_original_feature_space_%d.png' %data_input.idxs_tr[0])

    plotter_extra_sample2 = DataAndGatesPlotterDepthOne(model, data_input.x_tr[1])
    plotter_extra_sample2.plot_inverse_UMAP_transform_in_feature_space(umapper, data_input.untransformed_matched_x_tr[1], gate_data_idxs=gate_data_idxs, ms=5)
    plt.savefig('points_in_original_feature_space_%d.png' %data_input.idxs_tr[1])


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



def make_ori_features_plot(trained_model_path, params_path, trained_ummaper_path, trackers_path, savename='points_in_original_feature_space.png'):
    trained_model, trained_ummaper, data_input = load_results(trained_model_path, params_path, trained_ummaper_path, trackers_path)
    plotter = DataAndGatesPlotterDepthOne(trained_model, np.concatenate(data_input.x_tr))
    plotter.plot_inverse_UMAP_transform_in_feature_space(trained_ummaper, np.concatenate(data_input.untransformed_matched_x_tr, axis=0), gate_data_idxs=[[2, 3], [0, 1], [5, 6], [10, 7]])
    plt.savefig(savename)
    plt.clf()

def load_results(trained_model_path, params_path, trained_umapper_path, trackers_path):

    with open(params_path, 'rb') as f:
        params = pickle.load(f)
    model_params = params['model_params']
    trained_model = load_saved_model(trained_model_path, model_params)

    with open(trained_umapper_path, 'rb') as f:
        trained_umapper = pickle.load(f)

    with open(trackers_path, 'rb') as f:
        trackers = pickle.load(f)
        tracker = trackers[-1] #get tracker at very end of training

    data_input = tracker.data_input
    return trained_model, trained_umapper, data_input

if __name__ == '__main__':
    path_to_params = '../output/umap_with_feat_diff/params.pkl'
#    test_plot(path_to_params)

    path_to_model = '../output/umap_with_feat_diff/model.pkl'
    path_to_umapper = '../output/umap_with_feat_diff/transformer.pkl'
    path_to_trackers = '../output/umap_with_feat_diff/trackers.pkl'
    #make_ori_features_plot(path_to_model, path_to_params, path_to_umapper, path_to_trackers, savename='ori_plot_with_feat_diff_reg.png')
    test_plot('../configs/umap_with_feat_diff_reg.yaml')
