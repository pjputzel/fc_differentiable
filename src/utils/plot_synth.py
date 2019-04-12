import matplotlib.pyplot as plt
import numpy as np
import pickle
import sys
sys.path.append('../')


def plot_results(results_dict, params_dict):
    losses = results_dict['losses']
    accs = results_dict['accs']
    epoch_step_sz = params_dict['NUM_EPOCHS_PER_EVALUATION']
    iterations = [epoch_step_sz  * i for i in range(params_dict['n_epoch']//epoch_step_sz)]
    fig, axes = plt.subplots(2)
    axes[0].plot(iterations, losses)
    axes[1].plot(iterations, accs)

    plt.show()

if __name__ == '__main__':
    with open('../../output/synth/full_batch/batch_size=full_batch_scale=.4_te-tr=1000_1000.pkl', 'rb') as f:
        results_dict, params_dict = pickle.load(f)
    print('Total Training Time: ', results_dict['training_time']/60)
    plot_results(results_dict, params_dict)

    print('init gates: ', results_dict['root_init_gate'], results_dict['leaf_gate_init']) 
