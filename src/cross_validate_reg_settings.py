from main_UMAP_with_repeated_gates_and_clustering import main

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

def cross_validate_reg_settings(path_to_params):

    start_time = time.time()

    params = TransformParameterParser(path_to_params).parse_params()
    print(params)
    # TODO
    check_consistency_of_params(params)
    
    feat_diffs = params['cross_validate']['feat_diffs']
    neg_box_regs = params['cross_validate']['neg_box_regs']
    n_runs = params['cross_validate']['n_runs']
    avg_te_losses = {}
    avg_te_losses['feat_diff_vals'] = feat_diffs
    avg_te_losses['neg_box_reg_vals'] = neg_box_regs
    for feat_diff in feat_diffs:
        avg_te_losses[feat_diff] = {}
        for neg_box_reg in neg_box_regs:
            print('feat diff %.3f, neg_box_reg %.3f' %(feat_diff, neg_box_reg))
            params['model_params']['feature_diff_penalty'] = feat_diff
            params['model_params']['negative_box_penalty'] = neg_box_reg
            te_losses = []
            te_accs = []
            for i in range(n_runs):
                params['random_seed'] = i
                tracker = main(params)
                te_losses.append(tracker.metrics['te_log_loss'][-1].detach().cpu().numpy())
                te_accs.append(tracker.metrics['te_acc'][-1])

            avg_te_losses[feat_diff][neg_box_reg] = {\
                'te_log_loss': np.mean(np.array(te_losses)),
                'te_acc': np.mean(np.array(te_accs))
            }
    
    with open(os.path.join(params['save_dir'], 'avg_te_losses.pkl'), 'wb') as f:
        pickle.dump(avg_te_losses, f)    


def check_consistency_of_params(params):
    if params['train_params']['descent_type'] == 'joint_descent':
        if not params['train_params']['learning_rate_gates'] == params['train_params']['learning_rate_classifier']:
            raise ValueError('For joint descent learning rate gates and learning rate classifier must be equal')
    if params['train_params']['conv_thresh']:
        if params['train_params']['n_epoch']:
            warnings.warn('n_epoch parameter is not used when a conv_thresh is set. Training will continue until the change in loss is less than conv_thresh regardless of the number of epochs.')
    if params['transform_params']['use_labels_to_transform_data'] and not (params['transform_params']['transform_type'] == 'umap'):
        raise ValueError('Supervised data transformation only supported with umap')

if __name__ == '__main__':
    path_to_params = '../configs/umap_reg_cv.yaml'
    cross_validate_reg_settings(path_to_params)
