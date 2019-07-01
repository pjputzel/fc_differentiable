import csv
import os

import yaml

from train import *
from utils.bayes_gate import ModelTree
from utils.input import *

default_hparams = {
    'logistic_k': 100,
    'logistic_k_dafi': 1000,
    'regularization_penalty': 0,
    'negative_box_penalty': 0.1,
    'positive_box_penalty': 0.0,
    'corner_penalty': 1.0,
    'gate_size_penalty': 1.0,
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
    'seven_epochs_for_gate_motion_plot': [0, 100, 300, 500, 700, 900, 999],
    'test_size': 0.20,
    'experiment_name': 'default',
    'random_state': 123,
    'n_run': 2,
    'train_alternate': True,
    'output': {
        'type': 'full'
    }
}


def run_single_panel(hparams, random_state_start=0, model_checkpoint=True):

    if not os.path.exists('../output/%s' % hparams['experiment_name']):
        os.makedirs('../output/%s' % hparams['experiment_name'])
    with open('../output/%s/hparams.csv' % hparams['experiment_name'], 'w') as outfile:
        writer = csv.writer(outfile)
        for key, val in hparams.items():
            writer.writerow([key, val])

    cll_4d_input = Cll4d1pInput(hparams)

    for random_state in range(random_state_start, hparams['n_run']):
        hparams['random_state'] = random_state
        cll_4d_input.split(random_state)

        model_tree = ModelTree(cll_4d_input.reference_tree,
                               logistic_k=hparams['logistic_k'],
                               regularisation_penalty=hparams['regularization_penalty'],
                               negative_box_penalty=hparams['negative_box_penalty'],
                               positive_box_penalty=hparams['positive_box_penalty'],
                               corner_penalty=hparams['corner_penalty'],
                               gate_size_penalty=hparams['gate_size_penalty'],
                               init_tree=cll_4d_input.init_tree,
                               loss_type=hparams['loss_type'],
                               gate_size_default=hparams['gate_size_default'])

        dafi_tree = ModelTree(cll_4d_input.reference_tree,
                              logistic_k=hparams['logistic_k_dafi'],
                              negative_box_penalty=hparams['negative_box_penalty'],
                              positive_box_penalty=hparams['positive_box_penalty'],
                              corner_penalty=hparams['corner_penalty'],
                              gate_size_penalty=hparams['gate_size_penalty'],
                              init_tree=None,
                              loss_type=hparams['loss_type'],
                              gate_size_default=hparams['gate_size_default'])

        # dafi_tree = run_train_dafi(dafi_tree, hparams, cll_4d_input)
        model_tree, train_tracker, eval_tracker, run_time, model_checkpoint_dict = \
            run_train_model(model_tree, hparams, cll_4d_input, model_checkpoint=model_checkpoint)
        if hparams['output']['type'] == 'full':
            output_metric_dict = run_output(
                model_tree, dafi_tree, hparams, cll_4d_input, train_tracker, eval_tracker, run_time)
        elif hparams['output']['type'] == 'lightweight':
            output_metric_dict = run_lightweight_output_no_split_no_dafi(
                model_tree, dafi_tree, hparams, cll_4d_input, train_tracker, eval_tracker, run_time)
        else:
            raise ValueError('Output type not recognized')
            
        # only plot once
        # # if not os.path.isfile('../output/%s/metrics.png' % hparams['experiment_name']) and plot_and_write_output:
        # run_plot_metric(hparams, train_tracker, eval_tracker, dafi_tree, cll_4d_input, output_metric_dict)
        # run_plot_gates(hparams, train_tracker, eval_tracker, model_tree, dafi_tree, cll_4d_input)
        run_write_prediction(model_tree, dafi_tree, cll_4d_input, hparams)
        run_gate_motion_1p(hparams, cll_4d_input, model_checkpoint_dict)
        # model_checkpoint = False


if __name__ == '__main__':
    # run_single_panel(sys.argv[1], int(sys.argv[2]), True)
    hparams = default_hparams
#    yaml_filename = '../configs/cll_4d_1p_reg_grid_srch.yaml'
    yaml_filename = '../configs/cll_4d_1p_default.yaml'
    with open(yaml_filename, "r") as f_in:
        yaml_params = yaml.safe_load(f_in)
    hparams.update(yaml_params)
    hparams['init_method'] = "dafi_init" if hparams['dafi_init'] else "random_init"
    if hparams['train_alternate']:
        hparams['n_epoch_dafi'] = hparams['n_epoch'] // hparams['n_mini_batch_update_gates'] * (
                hparams['n_mini_batch_update_gates'] - 1)
    else:
        hparams['n_epoch_dafi'] = hparams['n_epoch']

    run_single_panel(hparams, 1, True)
    #Change this two lines to run in parallel
    
    #grid_neg_box = [1.]
    #grid_corner_reg = [1., 0., 0.1]

    #
    #grid_gate_size = [0.1]
    ##run_single_panel(hparams, 1, True)
    #for neg_box_reg in grid_neg_box:
    #    for corner_reg in grid_corner_reg:
    #        for gate_size_reg in grid_gate_size:
    #            hparams['negative_box_penalty'] = neg_box_reg
    #            hparams['corner_penalty'] = corner_reg
    #            hparams['gate_size_penalty'] = gate_size_reg
    #            hparams['experiment_name'] = 'grid_search_neg_box=%.2f_corner=%.2f_gate_size=%.2f' %(neg_box_reg, corner_reg, gate_size_reg)
    #            print(hparams)
    #            run_single_panel(hparams, 1, True)
