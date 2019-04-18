import matplotlib.pyplot as plt
import numpy as np
import pickle
import sys
import load_data as dh
import torch.nn.functional as F
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

def filter_gate(sample, gate):
    idxs = (sample[:, 0] > gate[0]) & (sample[:, 0] < gate[1]) & (sample[:, 1] > gate[2]) & (sample[:, 1] < gate[3])
    return sample[idxs]

def plot_gates(results_dict, params_dict, pos_normalized_sample,neg_normalized_sample,  offset, scale):
    
    FEATURE2ID = {'M1':0, 'M2':1, 'M3':2, 'M4':3}
    nested_list = \
        [
            [[u'M1', 0.000, 1.500], [u'M2', 0.000, 1.500]],
            [
                [
                    [[u'M3', 0.000, 1.500], [u'M4', 0.000, 1.500]],
                    []
                ]
            ]
        ]

    
    nested_ref_gates = dh.normalize_nested_tree(nested_list, offset, scale, FEATURE2ID)
    ref_gates = [[nested_ref_gates[0][0][1], nested_ref_gates[0][0][2], nested_ref_gates[0][1][1], nested_ref_gates[0][1][2]], [nested_ref_gates[1][0][0][0][1], nested_ref_gates[1][0][0][0][2], nested_ref_gates[1][0][0][1][1], nested_ref_gates[1][0][0][1][2]] ]

    leaf_i = results_dict['leaf_gate_init']
    leaf_gate_i = [F.sigmoid(leaf_i.gate_low1_param).item(), F.sigmoid(leaf_i.gate_upp1_param).item(), F.sigmoid(leaf_i.gate_low2_param).item(), F.sigmoid(leaf_i.gate_upp2_param).item()]


    root_i = results_dict['root_init_gate']
    root_gate_i = [F.sigmoid(root_i.gate_low1_param).item(), F.sigmoid(root_i.gate_upp1_param).item(), F.sigmoid(root_i.gate_low2_param).item(), F.sigmoid(root_i.gate_upp2_param).item()]

    leaf_f = results_dict['learned_leaf_gate']
    leaf_gate_f = [F.sigmoid(leaf_f.gate_low1_param).item(), F.sigmoid(leaf_f.gate_upp1_param).item(), F.sigmoid(leaf_f.gate_low2_param).item(), F.sigmoid(leaf_f.gate_upp2_param).item()]

    root_f = results_dict['learned_root_gate']
    root_gate_f = [F.sigmoid(root_f.gate_low1_param).item(), F.sigmoid(root_f.gate_upp1_param).item(), F.sigmoid(root_f.gate_low2_param).item(), F.sigmoid(root_f.gate_upp2_param).item()]


    print(leaf_gate_i)
    print(leaf_gate_f)
    print(ref_gates[1])
   
    print(root_gate_i)
    print(root_gate_f)
    print(ref_gates[0])


    size = 10
    fig, axes = plt.subplots(2, 2)
    axes[0][0].scatter(pos_normalized_sample[:, 0], pos_normalized_sample[:, 1], s=size)
    within_gate_pos = filter_gate(pos_normalized_sample, root_gate_f)
    axes[1][0].scatter(within_gate_pos[:, 2], within_gate_pos[:, 3], s=size)
    plot_gate(axes[0][0], root_gate_i, 'g', 'init_gate', dashed=True)
    plot_gate(axes[0][0], root_gate_f, 'g', 'final_gate')
    plot_gate(axes[0][0], ref_gates[0], 'b', 'by_inspection')
    axes[0][0].legend()

    plot_gate(axes[1][0], leaf_gate_i, 'g', 'init_gate', dashed=True)
    plot_gate(axes[1][0], leaf_gate_f, 'g', 'learned_gate')
    plot_gate(axes[1][0], ref_gates[1], 'b', 'by_inspection')
    axes[1][0].legend()


    axes[0][1].scatter(neg_normalized_sample[:, 0], neg_normalized_sample[:, 1], s=size)
    within_gate_neg = filter_gate(neg_normalized_sample, root_gate_f)
    axes[1][1].scatter(within_gate_neg[:, 2], within_gate_neg[:, 3], s=size)
    plot_gate(axes[0][1], root_gate_i, 'g', 'init_gate', dashed=True)
    plot_gate(axes[0][1], root_gate_f, 'g', 'final_gate')
    plot_gate(axes[0][1], ref_gates[0], 'b', 'by_inspection')
    axes[0][1].legend()

    plot_gate(axes[1][1], leaf_gate_i, 'g', 'init_gate', dashed=True)
    plot_gate(axes[1][1], leaf_gate_f, 'g', 'learned_gate')
    plot_gate(axes[1][1], ref_gates[1], 'b', 'by_inspection')
    axes[1][1].legend()
    plt.show()
    
def plot_box(axes, x1, x2, y1, y2, color, label, dashed=False):
    dash = [3,1]
    if dashed:
        axes.plot([x1, x1], [y1, y2], c=color, label=label, dashes=dash)
        axes.plot([x1, x2], [y1, y1], c=color, dashes=dash)
        axes.plot([x2, x2], [y1, y2], c=color, dashes=dash)
        axes.plot([x2, x1], [y2,y2], c=color, dashes=dash)
    else:
        axes.plot([x1, x1], [y1, y2], c=color, label=label)
        axes.plot([x1, x2], [y1, y1], c=color)
        axes.plot([x2, x2], [y1, y2], c=color)
        axes.plot([x2, x1], [y2,y2], c=color)
    return axes

def plot_gate(axes, gate, color, label, dashed=False):
    plot_box(axes, gate[0], gate[1], gate[2], gate[3], color, label, dashed=dashed)

    
if __name__== '__main__':
    with open('../../output/synth/batch_size=full_batchbatch_size=full_batch_scale=.4_te-tr=1000_100.pkl', 'rb') as f:
        results_dict, params_dict = pickle.load(f)
    SYNTH_DATA_DIR = '../../data/synth/'
    DATASET_NAME = 'synthex_scale=.4_N=2000.pkl'
    seed = 0
    np.random.seed(seed)
    with open(SYNTH_DATA_DIR + DATASET_NAME, 'rb') as f:
        samples, labels = pickle.load(f)
        indices = np.random.randint(0, len(samples), len(samples))
        samples = [samples[i] for i in indices]
        labels = [labels[i] for i in indices]
        print(labels)
        normalized_samples, offset, scale = dh.normalize_x_list(samples)

        FEATURE2ID = {'M1':0, 'M2':1, 'M3':2, 'M4':3}
    #print('Total Training Time: ', results_dict['training_time']/60)
    #plot_results(results_dict, params_dict)

    #print('init gates: ', results_dict['root_init_gate'], results_dict['leaf_gate_init']) 
    print( labels[0])
    print(labels[3])
    plot_gates(results_dict, params_dict, normalized_samples[1], normalized_samples[2], offset, scale)
