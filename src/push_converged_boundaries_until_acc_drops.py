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

def set_random_seeds(params):
    torch.manual_seed(params['random_seed'])
    np.random.seed(params['random_seed'])

def cross_validate_accuracy_over_saved_results(path_to_results, stepsize, n_steps, nfolds=20, starting_fold=30):
    path_to_params = os.path.join(path_to_results, 'params.yaml')

    params = TransformParameterParser(path_to_params).parse_params()
    print(params)
    cur_params = deepcopy(params)

    #evauntually uncomment this leaving asis in order ot keep the same results as before to compare.
    set_random_seeds(params)
    data_input = DataInput(params['data_params'])
    te_accs = []
    pushed_gates_per_fold = []
    starting_gates_per_fold = []
    diffs_per_fold = []

    for fold in range(starting_fold):
        data_input.split_data()

    for fold in range(starting_fold, nfolds + starting_fold):
        print('Running fold %d' %fold)
        cur_params['save_dir'] = os.path.join(params['save_dir'], 'run%d' %fold)
        data_input.split_data()
        best_tr_acc, starting_gate, best_gate = push_converged_boundaries_given_data_input_and_params(cur_params, data_input, stepsize, n_steps, path_to_params)

        model = DepthOneModel([[['D1', best_gate[0], best_gate[1]], ['D2', best_gate[2], best_gate[3]]]], params['model_params'])
        fit_classifier_params(model, data_input, params['train_params']['learning_rate_classifier'])
        te_acc = compute_te_acc(model, data_input)
        print('te acc for fold %d is %.3f' %(fold, te_acc))
        te_accs.append(te_acc)
        pushed_gates_per_fold.append(best_gate)
        starting_gates_per_fold.append(starting_gate)
        diffs_per_fold.append(get_diff_between_gates(starting_gate, best_gate))
        print('Diff: ', get_diff_between_gates(starting_gate, best_gate))


    print('Te accs:', te_accs)
    print('Diffs per fold:', diffs_per_fold)
    with open(os.path.join(path_to_results, 'expanded_boundaries_te_accs_per_fold.pkl'), 'wb') as f:
        pickle.dump(te_accs, f)
    with open(os.path.join(path_to_results, 'expanded_boundaries_diffs_per_fold.pkl'), 'wb') as f:
        pickle.dump(diffs_per_fold, f)
    with open(os.path.join(path_to_results, 'expanded_boundaries_best_pushed_gates_per_fold.pkl'), 'wb') as f:
        pickle.dump(pushed_gates_per_fold, f)
        
def get_diff_between_gates(gate1, gate2):
    return ((gate1[0] - gate2[0])**2 + (gate1[1] - gate2[1])**2 + (gate1[2] - gate2[2])**2 + (gate1[3] - gate2[3])**2 )**(1/2)


# Same as push converged boundaries after loading and setting seed
# just used for convenvience while cross validating.
def push_converged_boundaries_given_data_input_and_params(params, data_input, stepsize, n_steps, path_to_params):
    start_time = time.time()
    print('%d samples in the training data' %len(data_input.x_tr))

    with open(os.path.join(params['save_dir'], 'tracker.pkl'), 'rb') as f:
        tracker = pickle.load(f)

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

    init_acc = tracker.metrics['tr_acc'][-1]
    cur_best_acc = init_acc
    starting_gate = model.get_gates()[0]
    cur_gate = copy.deepcopy(starting_gate)
    cur_best_gate = copy.deepcopy(cur_gate)
    print('Starting gate:', starting_gate)
    counter = 0
    for left_step in range(n_steps):
        cur_gate[0] = starting_gate[0] - left_step * stepsize
        for right_step in range(n_steps):
            cur_gate[1] = starting_gate[1] + right_step * stepsize
            for down_step in range(n_steps):
                cur_gate[2] = starting_gate[2] - down_step * stepsize
                for up_step in range(n_steps):
                    cur_gate[3] = starting_gate[3] + up_step * stepsize
                    model = DepthOneModel([[['D1', cur_gate[0], cur_gate[1]], ['D2', cur_gate[2], cur_gate[3]]]], params['model_params'])
                    fit_classifier_params(model, data_input, params['train_params']['learning_rate_classifier'])
#                    model.nodes = None
#                    model.init_nodes([[['D1', cur_gate[0], cur_gate[1]], ['D2', cur_gate[2], cur_gate[3]]]])
                    cur_acc = compute_tr_acc(model, data_input)
                    #cur_acc = performance_tracker.metrics['tr_acc'][-1]
                    counter += 1
                    #print(counter)            
                    #print(cur_gate)
                    #print(cur_acc)
                    if cur_acc > cur_best_acc:
                        cur_best_acc = cur_acc
                        cur_best_gate = copy.deepcopy(cur_gate)
                
    print('Final acc %.3f, Initial acc %.3f' %(cur_best_acc, init_acc))
    print('Init/final gates', starting_gate, cur_best_gate)
    print('time taken: %d' %(time.time() - start_time))
    return cur_best_acc, starting_gate, cur_best_gate

'''
Takes a converged model and pushes the boundaries as far as possible without impacting the accuracy.
'''
def push_converged_boundaries(path_to_params, stepsize, n_steps):
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

    init_acc = trackers[0].metrics['tr_acc'][-1]
    cur_best_acc = init_acc
    starting_gate = model.get_gates()[0]
    cur_gate = copy.deepcopy(starting_gate)
    cur_best_gate = copy.deepcopy(cur_gate)
    print('Starting gate:', starting_gate)
    counter = 0
    for left_step in range(n_steps):
        cur_gate[0] = starting_gate[0] - left_step * stepsize
        for right_step in range(n_steps):
            cur_gate[1] = starting_gate[1] + right_step * stepsize
            for down_step in range(n_steps):
                cur_gate[2] = starting_gate[2] - down_step * stepsize
                for up_step in range(n_steps):
                    cur_gate[3] = starting_gate[3] + up_step * stepsize
                    model = DepthOneModel([[['D1', cur_gate[0], cur_gate[1]], ['D2', cur_gate[2], cur_gate[3]]]], params['model_params'])
                    fit_classifier_params(model, data_input, params['train_params']['learning_rate_classifier'])
#                    model.nodes = None
#                    model.init_nodes([[['D1', cur_gate[0], cur_gate[1]], ['D2', cur_gate[2], cur_gate[3]]]])
                    cur_acc = compute_tr_acc(model, data_input)
                    #cur_acc = performance_tracker.metrics['tr_acc'][-1]
                    counter += 1
                    print(counter)            
                    print(cur_gate)
                    print(cur_acc)
                    if cur_acc > cur_best_acc:
                        cur_best_acc = cur_acc
                        cur_best_gate = copy.deepcopy(cur_gate)
                
    print('Final acc %.3f, Initial acc %.3f' %(cur_best_acc, init_acc))
    print('Init/final gates', starting_gate, cur_best_gate)

def compute_tr_acc(model, data_input):
    tr_output = model(data_input.x_tr, data_input.y_tr) 
    y_true = data_input.y_tr
    y_pred = (tr_output['y_pred'].cpu().detach().numpy() >= 0.5) * 1.0
    y_pred = y_pred.reshape(y_true.cpu().numpy().shape)
    acc = sum(y_pred == y_true.cpu().numpy()) * 1.0 / y_true.shape[0]
    return acc

def compute_te_acc(model, data_input):
    te_output = model(data_input.x_te, data_input.y_te) 
    y_true = data_input.y_te
    y_pred = (te_output['y_pred'].cpu().detach().numpy() >= 0.5) * 1.0
    y_pred = y_pred.reshape(y_true.cpu().numpy().shape)
    acc = sum(y_pred == y_true.cpu().numpy()) * 1.0 / y_true.shape[0]
    return acc

if __name__ == '__main__':
    # be careful to make sure the results you want are saved in the savedir for the current params file
    ### single run
    #path_to_params = '../configs/umap_with_feat_diff_reg.yaml'
    #stepsize = .02
    #n_steps = 5
    #push_converged_boundaries(path_to_params, stepsize, n_steps)


    # Cross validation with saved results
    path_to_results = '../output/umap_with_feat_diff_cv_experiments/'
    stepsize = .033 # .033
    n_steps = 3 # 3
    cross_validate_accuracy_over_saved_results(path_to_results, stepsize, n_steps, nfolds=20, starting_fold=30)
