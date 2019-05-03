import csv
import os

import yaml

from train import *
from utils.bayes_gate import ModelForest
from utils.input import *

default_hparams = {
    'logistic_k': 100,
    'logistic_k_dafi': 1000,
    'regularization_penalty': 0,
    'negative_box_penalty': 0.1,
    'positive_box_penalty': 0,
    'corner_penalty': 10.0,
    'gate_size_penalty': 1,
    'gate_size_default': (0.5, 0.5),
    'load_from_pickle': True,
    'dafi_init': False,
    'optimizer': "Adam",  # or Adam, SGD
    'loss_type': 'logistic',  # or MSE
    'n_epoch_eval': 20,
    'n_mini_batch_update_gates': 50,
    'learning_rate_classifier': 0.05,
    'learning_rate_gates': 0.1,
    'batch_size': 10,
    'n_epoch': 1000,
    'test_size': 0.20,
    'experiment_name': 'default',
    'random_state': 123,
    'n_run': 1,
    'train_alternate': True,
}


def run_multiple_panel(yaml_filename, random_state_start=0, model_checkpoint=True):
    hparams = default_hparams
    with open(yaml_filename, "r") as f_in:
        yaml_params = yaml.safe_load(f_in)
    hparams.update(yaml_params)
    hparams['init_method'] = "dafi_init" if hparams['dafi_init'] else "random_init"
    hparams['n_epoch_dafi'] = hparams['n_epoch'] // hparams['n_mini_batch_update_gates'] * (
            hparams['n_mini_batch_update_gates'] - 1)
    print(hparams)

    if not os.path.exists('../output/%s' % hparams['experiment_name']):
        os.makedirs('../output/%s' % hparams['experiment_name'])
    with open('../output/%s/hparams.csv' % hparams['experiment_name'], 'w') as outfile:
        writer = csv.writer(outfile)
        for key, val in hparams.items():
            writer.writerow([key, val])

    cll_4d_2p_input = Cll4d2pInput(hparams)

    for random_state in range(random_state_start, hparams['n_run']):
        hparams['random_state'] = random_state
        cll_4d_2p_input.split(random_state)

        model_forest = ModelForest(cll_4d_2p_input.reference_tree,
                                   logistic_k=hparams['logistic_k'],
                                   regularisation_penalty=hparams['regularization_penalty'],
                                   negative_box_penalty=hparams['negative_box_penalty'],
                                   positive_box_penalty=hparams['positive_box_penalty'],
                                   corner_penalty=hparams['corner_penalty'],
                                   gate_size_penalty=hparams['gate_size_penalty'],
                                   init_tree_list=cll_4d_2p_input.init_tree,
                                   loss_type=hparams['loss_type'],
                                   gate_size_default=hparams['gate_size_default'])

        dafi_forest = ModelForest(cll_4d_2p_input.reference_tree,
                                  logistic_k=hparams['logistic_k_dafi'],
                                  regularisation_penalty=hparams['regularization_penalty'],
                                  negative_box_penalty=hparams['negative_box_penalty'],
                                  positive_box_penalty=hparams['positive_box_penalty'],
                                  corner_penalty=hparams['corner_penalty'],
                                  gate_size_penalty=hparams['gate_size_penalty'],
                                  init_tree_list=[None] * 2,
                                  loss_type=hparams['loss_type'],
                                  gate_size_default=hparams['gate_size_default'])

        dafi_forest = run_train_dafi(dafi_forest, hparams, cll_4d_2p_input)
        model_forest, train_tracker, eval_tracker, run_time, model_checkpoint_dict =\
            run_train_model(model_forest, hparams, cll_4d_2p_input, model_checkpoint=model_checkpoint)
        output_metric_dict = run_output(
            model_forest, dafi_forest, hparams, cll_4d_2p_input, train_tracker, eval_tracker, run_time)

        # only plot once
        if not os.path.isfile('../output/%s/metrics.png' % hparams['experiment_name']):
            run_plot_metric(hparams, train_tracker, eval_tracker, dafi_forest, cll_4d_2p_input, output_metric_dict)
        run_write_prediction(model_forest, dafi_forest, cll_4d_2p_input, hparams)
        run_gate_motion_in_one_figure(hparams, cll_4d_2p_input, model_checkpoint_dict)

#
if __name__ == '__main__':
    # run(sys.argv[1], int(sys.argv[2]))
    run_multiple_panel("../configs/cll_4d_2p_default.yaml", 0, True)
