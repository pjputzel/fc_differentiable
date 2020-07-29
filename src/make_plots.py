import pickle
import os
import matplotlib
import warnings
matplotlib.use('Agg')
matplotlib.rcParams['font.size'] = 10
matplotlib.rcParams['font.family'] = 'serif'

from utils.utils_plot import *
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import torch.nn.functional as F
import torch
from math import *
import utils.utils_load_data as dh
from utils.input import *
from expand_learned_cell_population import KDEGateExpander
from utils.utils_plot import run_leaf_gate_plots
from utils.utils_plot import run_gate_motion_from_saved_results
from utils.utils_plot import *
from utils.utils_plot_synth import *
from make_UMAP_embeddings import *
import yaml


from utils.DataInput import DataInput
from utils.DataAndGatesPlotter import DataAndGatesPlotterDepthOne

default_hparams = {
    'logistic_k': 100,
    'logistic_k_dafi': 10000,
    'regularization_penalty': 0,
    'negative_box_penalty': 0.0,
    'negative_proportion_default': 0.0001,
    'positive_box_penalty': 0.0,
    'corner_penalty': .0,
    'feature_diff_penalty': 0.,
    'gate_size_penalty': .0,
    'gate_size_default': (0.5, 0.5),
    'load_from_pickle': True,
    'dafi_init': False,
    'optimizer': "Adam",  # or Adam, SGD
    'loss_type': 'logistic',  # or MSE
    'n_epoch_eval': 100,
    'n_mini_batch_update_gates': 50,
    'learning_rate_classifier': 0.05,
    'learning_rate_gates': 0.05,
    'batch_size': 10,
    'n_epoch': 1000, 
    'seven_epochs_for_gate_motion_plot': [0, 50, 100, 200, 300, 400, 500],
    'test_size': 0.20,
    'experiment_name': 'default',
    'random_state': 123,
    'n_run': 2,
    'init_type': 'random_corner',
    'corner_init_deterministic_size': .75,
    'train_alternate': True,
    'run_logistic_to_convergence': False,
    'output': {
        'type': 'full'
    },
    'annealing': {
        'anneal_logistic_k': False,
        'final_k': 1000,
        'init_k': 1
    },
    'two_phase_training': {
        'turn_on': False,
        'num_only_log_loss_epochs': 50
    },
    'plot_params':{
        'figsize': [10, 10],
        'marker_size': .01,
    },
    'use_out_of_sample_eval_data': False,
}
def make_dev_data_plots():
    model_path = '../output/single_two_phase_gs=10/model.pkl'
    cell_sz = .1
    with open('../data/cll/x_dev_4d_1p.pkl', 'rb') as f:
        x_dev_list = pickle.load(f)

    with open('../data/cll/y_dev_4d_1p.pkl', 'rb') as f:
        labels= pickle.load(f)

    feature_names = ['CD5', 'CD19', 'CD10', 'CD79b']
    feature2id = dict((feature_names[i], i) for i in range(len(feature_names)))
    x_dev_list, offset, scale = dh.normalize_x_list(x_dev_list)

#    get_dafi_gates(offset, scale, feature2id)
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    DAFI_GATES = get_dafi_gates(offset, scale, feature2id)

    plot_samples_and_gates_cll_4d_dev(x_dev_list, labels, model, DAFI_GATES, cell_sz=cell_sz)

def make_accs_and_losses_final_model():
    savefolder = '../output/CV_neg=0.001_diff=0.001_FINAL_OOS_seed0/'
    tracker_train_path = savefolder + 'tracker_train_m.pkl'
    tracker_eval_path = savefolder + 'tracker_eval_m.pkl'

    plot_accs_and_losses(tracker_train_path, tracker_eval_path, savefolder=savefolder)

def make_synth_plot(hparams, model_paths, cells_to_plot=100000, device=1):
    with open(model_paths['init'], 'rb') as f:
        model_init = pickle.load(f).cuda(device)
    if hparams['dictionary_is_broken']:
        model_init.fix_children_dict_synth()
    
    with open(model_paths['final'], 'rb') as f:
        model_final = pickle.load(f).cuda(device)
    if hparams['dictionary_is_broken']:
        model_final.fix_children_dict_synth()

    models = {'init': model_init, 'final': model_final}

    synth_input = SynthInput(hparams, device=device)
    data = [x.cpu().detach().numpy() for x in synth_input.x_train]
    data_pos = [x for x, y in zip(data, synth_input.y_train) if y == 1.]
    data_neg = [x for x, y in zip(data, synth_input.y_train) if y == 0.]
    catted_data_pos = np.concatenate(data_pos)
    shuffled_idxs = np.random.permutation(len(catted_data_pos))
    print('max is', np.max(catted_data_pos[shuffled_idxs][0:cells_to_plot]))
    plot_synth_data_with_gates(models, catted_data_pos[shuffled_idxs][0:cells_to_plot], hparams, {'title': 'Class 1 Results'})
    savepath = '../output/%s/synth_gates_pos.png' %hparams['experiment_name']
    plt.tight_layout()
    plt.savefig(savepath, dpi=300, bbox_inches='tight')
    plt.clf()

    catted_data_neg = np.concatenate(data_neg)
    shuffled_idxs = np.random.permutation(len(catted_data_neg))
    plot_synth_data_with_gates(models, catted_data_neg[shuffled_idxs][0:cells_to_plot], hparams, {'title': 'Class 2 Results'})
    savepath = '../output/%s/synth_gates_neg.png' %hparams['experiment_name']
    plt.tight_layout()
    plt.savefig(savepath, dpi=300, bbox_inches='tight')
    plt.clf()

def make_model_plot(hparams, path_to_model, device):
    # make a run that takes in just a model rather than a model chekcpoint dict
    run_model_single_iter_pos_and_neg_gates(hparams, path_to_model, device_data=device)

def make_model_plots_both_panels(hparams, path_to_model_checkpoints, num_iters=120):
    with open(path_to_model_checkpoints, 'rb') as f:
        model_checkpoints = pickle.load(f)
    model_init = model_checkpoints[0]
    model_final = model_checkpoints[num_iters]
    run_both_panels_pos_and_neg_gates(model_init, hparams, savename='pos_and_neg_plots_both_init.png')
    run_both_panels_pos_and_neg_gates(model_final, hparams, savename='pos_and_neg_plots_both_final.png')


def make_dafi_plot(hparams):
    run_dafi_single_iter_pos_and_neg_gates(hparams, device_data=0)


def make_model_loss_plots(output_dir, figsize=(9, 3)):
    with open(os.path.join(output_dir, 'tracker_train_m.pkl'),'rb') as f:
            tracker_train = pickle.load(f)
    with open(os.path.join(output_dir, 'tracker_eval_m.pkl'), 'rb') as f:
            tracker_eval = pickle.load(f)
    log_loss_train = tracker_train.log_loss
    log_loss_eval = tracker_eval.log_loss

    acc_train = tracker_train.acc
    acc_eval = tracker_eval.acc

    reg_train = [n + f for n, f in zip(tracker_train.neg_prop_loss, tracker_train.feature_diff_loss)]
    reg_eval = [n + f for n, f in zip(tracker_eval.neg_prop_loss, tracker_eval.feature_diff_loss)]

    fig, axes = plt.subplots(1, 3, figsize=figsize)
    x_ticks = np.arange(len(log_loss_train))
    axes[0].plot(x_ticks, acc_train, color='b', label='Train')
    axes[1].plot(x_ticks, log_loss_train, color='b', label='Train')
    axes[2].plot(x_ticks, reg_train, color='b', label='Train')

    axes[0].plot(x_ticks, acc_eval, color='tab:orange', label='Test')
    axes[1].plot(x_ticks, log_loss_eval, color='tab:orange', label='Test')
    axes[2].plot(x_ticks, reg_eval, color='tab:orange', label='Test')
    savepath = os.path.join(output_dir, 'diagnostics.png')
    plt.tight_layout()
    plt.savefig(savepath, dpi=300, bbox_inches='tight')

def make_umap_plot(embeddings_with_labels):

    embeddings_with_labels = np.concatenate(embeddings_with_labels)
    positive_samples = embeddings_with_labels[embeddings_with_labels[:, 2] == 1]
    negative_samples = embeddings_with_labels[embeddings_with_labels[:, 2] == 0]
    
    plt.scatter(positive_samples[:, 0], positive_samples[:, 1], color='r', s=.001, alpha=.1)
    plt.scatter(negative_samples[:, 0], negative_samples[:, 1], color='b', s=.001, alpha=.1)
    plt.savefig('Embeddings_with_labels_colored.png')

    plt.clf()
    plt.xlim(-18, -10)
    plt.ylim(-8, -1)
    plt.scatter(positive_samples[:, 0], positive_samples[:, 1], color='r', s=.001, alpha=.1)
    plt.scatter(negative_samples[:, 0], negative_samples[:, 1], color='b', s=.001, alpha=.1)
    plt.savefig('Embeddings_with_labels_colored_zoomed_in.png')
    

def make_umap_plot_and_embed_samples(umapper_path, data_path, labels_path):
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    with open(labels_path, 'rb') as f:
        labels = pickle.load(f)
    with open(umapper_path, 'rb') as f:
        umapper = pickle.load(f)
    embeddings_with_labels = np.concatenate(embed_samples_with_labels(umapper, data, labels))
    make_umap_plot(embeddings_with_labels)


def set_random_seeds(params):
    torch.manual_seed(params['random_seed'])
    np.random.seed(params['random_seed'])

def make_umap_plots_for_incorrect_and_correct_samples(
    results_path, plot_expanded_data=True, path_to_true_features=None,
    BALL=False):
    with open(os.path.join(results_path, 'configs.pkl'), 'rb') as f:
        params = pickle.load(f)

    with open(os.path.join(results_path, 'transformer.pkl'), 'rb') as f:
        umapper = pickle.load(f)

    sample_names_to_true_features = None
    if path_to_true_features:
        with open(path_to_true_features, 'rb') as f:
            sample_names_to_true_features = pickle.load(f)

    set_random_seeds(params)    

    model = DepthOneModel([[['D1', 0, 0], ['D2', 0, 0]]], params['model_params'])
    model.load_state_dict(torch.load(os.path.join(results_path, 'model.pkl')))
    try: 
        print(params['data_params']['use_presplit_data'])
    except:
        params['data_params']['use_presplit_data'] = False
    data_input = DataInput(params['data_params'])
    # splitting because codebase requires a split currently
    data_input.split_data()
    print('embedding data')
    # only for debuggin
    #params['transform_params']['cells_to_subsample'] = 2
    data_input.embed_data(
        umapper,
        cells_to_subsample = params['transform_params']['cells_to_subsample'],
        use_labels_to_transform_data = params['transform_params']['use_labels_to_transform_data']
    )

    data_input.normalize_data()
    data_input.convert_all_data_to_tensors()

    # gate expansion using kde
    if plot_expanded_data:
        print(model.get_gates()[0])
        kde_expander = KDEGateExpander(data_input.x_tr, model.get_gates()[0], sigma_thresh_factor=.5)
        kde_expander.expand_gates()
        kde_expander.collect_expanded_cells_per_sample()
        tr_expanded_data = kde_expander.expanded_data_per_sample
        te_expanded_data = kde_expander.get_expanded_data_new_samples(data_input.x_te)
    else:
        tr_expanded_data = None
        te_expanded_data = None
    output_tr = model(data_input.x_tr, data_input.y_tr)
    output_te = model(data_input.x_te, data_input.y_te)
    matching_tr = [( (output_tr['y_pred'].cpu().detach().numpy() >= .5)[i] * 1.0 == data_input.y_tr[i] ) for i in range(len(data_input.y_tr))]
    pos_probs_tr = np.array([prob.cpu().detach().numpy() for prob in output_tr['y_pred']])
    sorted_idxs_tr = np.argsort(pos_probs_tr)

    #correct_idxs_tr = [data_input.idxs_tr[i]  for i in range(len(data_input.y_tr)) if matching_tr[i]]
    correct_idxs_tr = [data_input.idxs_tr[i]  for i in sorted_idxs_tr if matching_tr[i]]

    correct_idxs_true_pos_tr = [idx for idx in correct_idxs_tr if data_input.y_tr[data_input.idxs_tr.index(idx)] == 1]
    correct_idxs_true_neg_tr = [idx for idx in correct_idxs_tr if data_input.y_tr[data_input.idxs_tr.index(idx)] == 0]

    #incorrect_idxs_tr = [data_input.idxs_tr[i]  for i in range(len(data_input.y_tr)) if not matching_tr[i]]
    incorrect_idxs_tr = [data_input.idxs_tr[i]  for i in sorted_idxs_tr if not matching_tr[i]]
    incorrect_idxs_true_pos_tr = [idx for idx in incorrect_idxs_tr if data_input.y_tr[data_input.idxs_tr.index(idx)] == 1]
    incorrect_idxs_true_neg_tr = [idx for idx in incorrect_idxs_tr if data_input.y_tr[data_input.idxs_tr.index(idx)] == 0]


    print(np.sum(correct_idxs_tr)/len(data_input.x_tr))

    matching_te = [( (output_te['y_pred'].cpu().detach().numpy() >= .5)[i] * 1.0 == data_input.y_te[i] ) for i in range(len(data_input.y_te))]
    pos_probs_te = np.array([prob.cpu().detach().numpy() for prob in output_te['y_pred']])
    sorted_idxs_te = np.argsort(pos_probs_te)

    #correct_idxs_te = [data_input.idxs_te[i]  for i in range(len(data_input.y_te)) if matching_te[i]]
    correct_idxs_te = [data_input.idxs_te[i]  for i in sorted_idxs_te if matching_te[i]]
    correct_idxs_true_pos_te = [idx for idx in correct_idxs_te if data_input.y_te[data_input.idxs_te.index(idx)] == 1]
    correct_idxs_true_neg_te = [idx for idx in correct_idxs_te if data_input.y_te[data_input.idxs_te.index(idx)] == 0]

    #incorrect_idxs_te = [data_input.idxs_te[i]  for i in range(len(data_input.y_te)) if not matching_te[i]]
    incorrect_idxs_te = [data_input.idxs_te[i]  for i in sorted_idxs_te if not matching_te[i]]
    incorrect_idxs_true_pos_te = [idx for idx in incorrect_idxs_te if data_input.y_te[data_input.idxs_te.index(idx)] == 1]
    incorrect_idxs_true_neg_te = [idx for idx in incorrect_idxs_te if data_input.y_te[data_input.idxs_te.index(idx)] == 0]
    print('correct te idxs:', correct_idxs_te, 'incorrect te idxs', incorrect_idxs_te)
    print(incorrect_idxs_true_neg_te)




    background_data_to_plot_neg = np.concatenate([data for i, data in enumerate(data_input.x_tr)  if data_input.y_tr[i] == 0])
    try:
        background_data_to_plot_neg = np.concatenate([background_data_to_plot_neg, np.concatenate([data for i, data in enumerate(data_input.x_te)  if data_input.y_te[i] == 0])])
    except:
        pass


    background_data_to_plot_pos = np.concatenate([data for i, data in enumerate(data_input.x_tr)  if data_input.y_tr[i]])
    background_data_to_plot_pos = np.concatenate([background_data_to_plot_pos, np.concatenate([data for i, data in enumerate(data_input.x_te)  if data_input.y_te[i]])])

    full_background_data_to_plot = np.concatenate([background_data_to_plot_pos, background_data_to_plot_neg])

    ### CHANGE SAVENAME IF YOU USE VAL DATA HERE
    plots_per_row_BALL = 9
    make_umap_plots_per_sample(model, data_input, incorrect_idxs_true_pos_tr, savename='true_pos_incorrect_dev_tr.png', plots_per_row=plots_per_row_BALL, background_data_to_plot=full_background_data_to_plot, expanded_data_per_sample=tr_expanded_data, sample_names_to_true_features=sample_names_to_true_features, BALL=BALL)
    make_umap_plots_per_sample(model, data_input, incorrect_idxs_true_neg_tr, savename='true_neg_incorrect_dev_tr.png', plots_per_row=plots_per_row_BALL, background_data_to_plot=full_background_data_to_plot, expanded_data_per_sample=tr_expanded_data, sample_names_to_true_features=sample_names_to_true_features, BALL=BALL)
    make_umap_plots_per_sample(model, data_input, correct_idxs_true_pos_tr, savename='true_pos_correct_dev_tr.png', plots_per_row=plots_per_row_BALL, background_data_to_plot=full_background_data_to_plot, expanded_data_per_sample=tr_expanded_data, sample_names_to_true_features=sample_names_to_true_features, BALL=BALL)
    make_umap_plots_per_sample(model, data_input, correct_idxs_true_neg_tr, savename='true_neg_correct_dev_tr.png', plots_per_row=plots_per_row_BALL, background_data_to_plot=full_background_data_to_plot, expanded_data_per_sample=tr_expanded_data, sample_names_to_true_features=sample_names_to_true_features, BALL=BALL)


    make_umap_plots_per_sample(model, data_input, incorrect_idxs_true_pos_te, savename='true_pos_incorrect_dev_te.png', plots_per_row=plots_per_row_BALL, background_data_to_plot=full_background_data_to_plot, expanded_data_per_sample=te_expanded_data, sample_names_to_true_features=sample_names_to_true_features, BALL=BALL)
    make_umap_plots_per_sample(model, data_input, incorrect_idxs_true_neg_te, savename='true_neg_incorrect_dev_te.png', plots_per_row=plots_per_row_BALL, background_data_to_plot=full_background_data_to_plot, expanded_data_per_sample=te_expanded_data, sample_names_to_true_features=sample_names_to_true_features, BALL=BALL)
    make_umap_plots_per_sample(model, data_input, correct_idxs_true_pos_te, savename='true_pos_correct_dev_te.png', plots_per_row=plots_per_row_BALL, background_data_to_plot=full_background_data_to_plot, expanded_data_per_sample=te_expanded_data, sample_names_to_true_features=sample_names_to_true_features, BALL=BALL)
    make_umap_plots_per_sample(model, data_input, correct_idxs_true_neg_te, savename='true_neg_correct_dev_te.png', plots_per_row=plots_per_row_BALL, background_data_to_plot=full_background_data_to_plot, expanded_data_per_sample=te_expanded_data, sample_names_to_true_features=sample_names_to_true_features, BALL=BALL)

#    make_umap_plots_per_sample(model, data_input, incorrect_idxs_te, savename='all_incorrect_dev_te.png')
#    make_umap_plots_per_sample(model, data_input, correct_idxs_tr, savename='all_correct_dev_tr.png')
#    
#    make_umap_plots_per_sample(model, data_input, correct_idxs_te, savename='all_correct_dev_te.png')
#
    
def make_umap_plots_per_sample(model, data_input, sample_idxs_to_plot, plots_per_row=5, figlen=7, savename='plots_per_sample.png', background_data_to_plot=None, color='b', expanded_data_per_sample=None, sample_names_to_true_features=None, BALL=False):


    if len(sample_idxs_to_plot) == 0:
        print('idxs are empty!')
        return None
    vals_to_delete = []
    for i, idx in enumerate(sample_idxs_to_plot):
        if not (idx in data_input.sample_names_all):
            print('Sample %d not in training data' %idx)
            vals_to_delete.append(idx)
    for val in vals_to_delete:
        del sample_idxs_to_plot[sample_idxs_to_plot.index(val)]
    plotter = DataAndGatesPlotterDepthOne(model, [])
    idxs_in_data_input = [[1, data_input.idxs_tr.index(idx)] if idx in data_input.idxs_tr else [0, data_input.idxs_te.index(idx)] for idx in sample_idxs_to_plot]

    n_samples_to_plot =  len(sample_idxs_to_plot)

    evenly_divides = not(n_samples_to_plot % plots_per_row)
    n_rows = n_samples_to_plot//plots_per_row
    if not(evenly_divides):
        n_rows += 1
    
    print(n_rows, plots_per_row) 
    fig, axes = plt.subplots(n_rows, plots_per_row, figsize=((figlen) * plots_per_row, (figlen) * n_rows),sharex=True, sharey=True)

    #fig.suptitle('UMAP Embedding and Learned Gates per Sample')
    
    
        
    axes = [axes] if len(axes.shape) == 1 else axes

    if not (background_data_to_plot is None):
        for i in range(n_rows):
            for j in range(plots_per_row):
                axes[i][j].scatter(
                    background_data_to_plot[:, 0],
                    background_data_to_plot[:, 1],
                    c='lightgrey', s=1/100, alpha=.5 
                )                      

    axes[0][0].set_xlim(0, 1)
    axes[0][0].set_ylim(0, 1)
    fig.tight_layout(pad=1.3)
    row_start_idx = 0
    for i, row in enumerate(range(n_rows)):
        sample_row_idxs = sample_idxs_to_plot[row_start_idx: row_start_idx + plots_per_row]
        for j, sample_idx in enumerate(sample_row_idxs):
            cur_axis = axes[i][j]
            data_input_matching_idx = idxs_in_data_input[row_start_idx + j][1]
            if idxs_in_data_input[j][0]:
                sample = data_input.x_tr[data_input_matching_idx]
                label = data_input.y_tr[data_input_matching_idx]
            else:
                sample = data_input.x_te[data_input_matching_idx]
                label = data_input.y_te[data_input_matching_idx]
            name = sample_idxs_to_plot[row_start_idx + j]
            true_feature = None
            if sample_names_to_true_features:
                true_feature = sample_names_to_true_features[name]
            if not BALL:
                plotter.plot_single_sample_with_gate(
                    sample, name, label,
                    cur_axis, size=1, color='b',
                    true_feature=true_feature
                )
            else:
                plotter.plot_single_sample_with_gate(
                    sample, name, label,
                    cur_axis, size=1, color='b',
                    true_feature=true_feature,
                    BALL=True
                )
            if not (expanded_data_per_sample is None):
                expanded_data = expanded_data_per_sample[data_input_matching_idx]
                cur_axis.scatter(expanded_data[:, 0], expanded_data[:, 1], s=1, color='r')
        row_start_idx += plots_per_row

 
    plt.savefig(savename)
    
        
    
    
    


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    
    ### for plotting a saved per sample umap embedding
    #per_sample_embedding_path = '../output/UMAP_embeddings/Per_Sample_Embeddings_num_neighbors=15_min_dist=0.10.pkl'
    #with open(per_sample_embedding_path, 'rb') as f:
    #    per_sample_embeddings = pickle.load(f)
    #print(per_sample_embeddings)
    #make_umap_plot(per_sample_embedding_path)
    
    sample_idxs = [26129, 26127, 26450, 25494, 26099]
    ### For UMAP combined plot with CLL
    #path = '../output/umap_with_feat_diff_newest/'

    ### For UMAP combined plot with B-ALL
    #path = '../output/umap_BALL/'
    path = '../output/umap_BALL_elliptical'
    path_to_true_features = '../data/B-ALL/easy_data_sample_names_to_true_features.pkl'
    make_umap_plots_for_incorrect_and_correct_samples(path, path_to_true_features=path_to_true_features, plot_expanded_data=False, BALL=True)

    #for quick umap plot
    #umapper_path= '../output/UMAP_embeddings/num_neighbors=15_min_dist=0.10.pkl_umapper.pkl'
    #data_path = '../data/cll/x_UMAP_dev_FIXED.pkl'
    #labels_path = '../data/cll/y_UMAP_dev.pkl'
    #make_umap_plots(umapper_path, data_path, labels_path)
    




    #experiment_yaml_file = '../configs/testing_corner_init.yaml'
    #experiment_yaml_file = '../configs/testing_overlaps.yaml'
    #experiment_yaml_file = '../configs/testing_my_heuristic_init.yaml'

    # for both panels plots
    #path_to_model_checkpoints = '../output/Both_Panels_CV_neg=0.001_diff=0.001_seed1/model_checkpoints.pkl'
    #yaml_filename = '../configs/both_panels.yaml'

    # for dafi/model plots
    #path_to_saved_model = '../output/FINAL_MODEL_neg=0.001_diff=0.001_seed0/init_model.pkl'
    #yaml_filename = '../configs/OOS_Final_Model.yaml'
    model_paths_synth =\
        {
            'init': '../output/Synth_same_reg_as_alg_seed0/model_init_seed0.pkl',
            'final': '../output/Synth_same_reg_as_alg_seed0/model_final_seed0.pkl'
        }

    #make_accs_and_losses_final_model()

    # for synth plots
    yaml_filename = '../configs/synth_plot.yaml'

    # for dafi/model plots
    #path_to_saved_model = '../output/FINAL_MODEL_neg=0.001_diff=0.001_seed0/init_model.pkl'
    #path_to_saved_model = '../output/Middle_neg=0.001_diff=0.001_FINAL_OOS_seed0/model.pkl'
    #yaml_filename = '../configs/OOS_Final_Model.yaml'
    #yaml_filename = '../configs/FINAL_MODEL_middle_init.yaml'
    



    #hparams = default_hparams
    #with open(yaml_filename, "r") as f_in:
    #    yaml_params = yaml.safe_load(f_in)
    #hparams.update(yaml_params)

    #make_dafi_plot(hparams)
    #make_synth_plot(hparams, model_paths_synth)
    #make_model_plot(hparams, path_to_saved_model, 0)
    #make_model_loss_plots('../output/CV_neg=0.001_diff=0.001_FINAL_OOS_seed0')
    #make_model_plots_both_panels(hparams, path_to_model_checkpoints)


    #run_gate_motion_from_saved_results(experiment_yaml_file)
#    run_leaf_gate_plots(experiment_yaml_file)
    #run_single_iter_pos_and_neg_gates_plot(experiment_yaml_file)
    #make_dev_data_plots(hparams)
