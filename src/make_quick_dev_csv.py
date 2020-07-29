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
from sklearn.cluster import KMeans
import torch
import os
import time
from copy import deepcopy
import utils.utils_load_data as dh


def main(path_to_params):
    params = TransformParameterParser(path_to_params).parse_params()
    print(params)
    data_input = load_and_prepare_data_input(params)

    
    # concatenate the labels and the sample names per sample first
    samples_with_labels_and_names = []
    for i, sample in enumerate(data_input.x_tr):
        cell_labels = np.ones([sample.shape[0], 1]) * int(data_input.y_tr[i])
        cell_sample_names = np.array([data_input.idxs_tr[i]] * int(sample.shape[0])).reshape(-1, 1)
        new_sample = np.concatenate([sample, cell_labels], axis=1)
        new_sample = np.concatenate([new_sample, cell_sample_names], axis=1)
        samples_with_labels_and_names.append(new_sample)

    # then concatenate the cells in each sample
    all_cells = np.concatenate(samples_with_labels_and_names)
    
    np.savetxt('cell_by_cell_dev.csv', all_cells, delimiter=',', fmt='%.5f')







def load_and_prepare_data_input(params):
    data_input = DataInput(params['data_params'])
    data_input.split_data()
    print('%d samples in the training data' %len(data_input.x_tr))

    with open(os.path.join(params['save_dir'], 'transformer.pkl'), 'rb') as f:
        data_transformer = pickle.load(f)

    # for debugging
    #params['transform_params']['cells_to_subsample'] = 2
    data_input.embed_data(\
        data_transformer,
        cells_to_subsample=params['transform_params']['cells_to_subsample'],
        use_labels_to_transform_data=params['transform_params']['use_labels_to_transform_data']
    ) 
    data_input.normalize_data()
    data_input.convert_all_data_to_tensors()
    return data_input

if __name__ == '__main__':
    path_to_params = '../configs/umap_with_presplit.yaml'
    main(path_to_params)
