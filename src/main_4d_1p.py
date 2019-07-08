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
    'seven_epochs_for_gate_motion_plot': [0, 50, 100, 200, 300, 400, 500],
    'test_size': 0.20,
    'experiment_name': 'default',
    'random_state': 123,
    'n_run': 2,
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
        if hparams['two_phase_training']['turn_on']: 
            model_tree, train_tracker, eval_tracker, run_time, model_checkpoint_dict = \
                run_train_model_two_phase(hparams, cll_4d_input, model_checkpoint=model_checkpoint)
        else:
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
    #all of this needs to be put into a parser object for the hparams
    # run_single_panel(sys.argv[1], int(sys.argv[2]), True)
    hparams = default_hparams
    #yaml_filename = '../configs/cll_4d_1p_reg_grid_srch.yaml'
    #yaml_filename = '../configs/testing_log_to_conv.yaml'
    #yaml_filename = '../configs/log_to_conv_gridsrch.yaml'
    #yaml_filename = '../configs/testing_two_phase_training.yaml'
    #yaml_filename = '../configs/two_phase_grid_search.yaml'
    yaml_filename = '../configs/single_two_phase.yaml'













    with open(yaml_filename, "r") as f_in:
        yaml_params = yaml.safe_load(f_in)
    hparams.update(yaml_params)
    hparams['init_method'] = "dafi_init" if hparams['dafi_init'] else "random_init"
    if hparams['train_alternate']:
        hparams['n_epoch_dafi'] = hparams['n_epoch'] // hparams['n_mini_batch_update_gates'] * (
                hparams['n_mini_batch_update_gates'] - 1)
    else:
        hparams['n_epoch_dafi'] = hparams['n_epoch']
    #while True:    
    #    run_single_panel(hparams, 1, True)
    

    run_single_panel(hparams, 1, True)





###### Two phase grid search for use with two phase grid search yaml file
    #grid_gate_size = [0., .25, .5, .75, 1., 1.25, 1.5, 1.75, 2., 10.]
#    grid_gate_size = [10., 2., 1.75, 1.5, 1.25, 1., .75, .5, .25, 0.]
#   
#    for gate_size_reg in grid_gate_size:
#        print('Gate size reg %.2f' %gate_size_reg)
#        hparams['gate_size_penalty'] = gate_size_reg
# 
#        hparams['experiment_name'] = 'two_phase_logreg_to_conv_grid_search_gate_size=%.2f' %(gate_size_reg)
#        run_single_panel(hparams, 1, True)


    ###Old grid srch code
    #Change this two lines to run in parallel

    
#    grid_neg_box = [0.]

#    grid_corner_reg = [0.001, 0.01, 0.05]
#
#    
#    grid_gate_size = [0.25, 0.5]
#    #run_single_panel(hparams, 1, True)
#    while True:
#        for corner_reg in grid_corner_reg:
#            for gate_size_reg in grid_gate_size:
#                    
#                    #for neg_box_reg in grid_neg_box:
#                        #hparams['negative_box_penalty'] = neg_box_reg
#                        #hparams['experiment_name'] = 'logreg_to_conv_grid_search_neg_box=%.2f_corner=%.2f_gate_size=%.2f' %(neg_box_reg, corner_reg, gate_size_reg)
#                hparams['corner_penalty'] = corner_reg
#                hparams['gate_size_penalty'] = gate_size_reg
#                hparams['experiment_name'] = 'logreg_to_conv_grid_search_corner=%.3f_gate_size=%.3f_default_gs=.25' %(corner_reg, gate_size_reg)
#                print(hparams)
#                run_single_panel(hparams, 1, True)
