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
    for i in range(len(results_dict['gates_per_iter'])):
        plot_single_gate(results_dict, ref_gates, pos_normalized_sample, neg_normalized_sample, i)


def plot_single_gate(results_dict, ref_gates, pos_normalized_sample, neg_normalized_sample, i, is_old_format=False):
        leaf_i = results_dict['leaf_gate_init']
        leaf_gate_i = [F.sigmoid(leaf_i.gate_low1_param).item(), F.sigmoid(leaf_i.gate_upp1_param).item(), F.sigmoid(leaf_i.gate_low2_param).item(), F.sigmoid(leaf_i.gate_upp2_param).item()]


        root_i = results_dict['root_init_gate']
        root_gate_i = [F.sigmoid(root_i.gate_low1_param).item(), F.sigmoid(root_i.gate_upp1_param).item(), F.sigmoid(root_i.gate_low2_param).item(), F.sigmoid(root_i.gate_upp2_param).item()]

        if is_old_format:
            leaf_f = results_dict['learned_leaf_gate']
        else:
            leaf_f = results_dict['gates_per_iter'][i][1]
        leaf_gate_f = [F.sigmoid(leaf_f.gate_low1_param).item(), F.sigmoid(leaf_f.gate_upp1_param).item(), F.sigmoid(leaf_f.gate_low2_param).item(), F.sigmoid(leaf_f.gate_upp2_param).item()]

        if is_old_format:
            root_f = results_dict['learned_root_gate']
        else:
            root_f = results_dict['gates_per_iter'][i][0]
        root_gate_f = [F.sigmoid(root_f.gate_low1_param).item(), F.sigmoid(root_f.gate_upp1_param).item(), F.sigmoid(root_f.gate_low2_param).item(), F.sigmoid(root_f.gate_upp2_param).item()]


        #print(leaf_gate_i)
        #print(leaf_gate_f)
        #print(ref_gates[1])
   
        #print(root_gate_i)
        #print(root_gate_f)
        #print(ref_gates[0])


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
        mng = plt.get_current_fig_manager()
        mng.full_screen_toggle()
        plt.show()

def make_motion_plot(results_dict, pos_normalized_sample, neg_normalized_sample, iters_idxs, is_old_format=False):
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
        num_dims = 2 #hardcoded for four right now
        num_classes = 2
        fig, axes = plt.subplots(num_dims * num_classes, len(iters_idxs))
        for idx, i in enumerate(iters_idxs):
            pos_axes = [axes[0][idx], axes[1][idx]]
            neg_axes = [axes[2][idx], axes[3][idx]]

            plot_single_iter_on_axes(results_dict, ref_gates, pos_axes, neg_axes, pos_normalized_sample, neg_normalized_sample, i, is_old_format=False)
        mng = plt.get_current_fig_manager()
        mng.full_screen_toggle()
        plt.show()
def plot_single_iter_on_axes(results_dict, ref_gates, pos_axes, neg_axes, pos_normalized_sample, neg_normalized_sample, i, is_old_format=False):
        leaf_i = results_dict['leaf_gate_init']
        leaf_gate_i = [F.sigmoid(leaf_i.gate_low1_param).item(), F.sigmoid(leaf_i.gate_upp1_param).item(), F.sigmoid(leaf_i.gate_low2_param).item(), F.sigmoid(leaf_i.gate_upp2_param).item()]


        root_i = results_dict['root_init_gate']
        root_gate_i = [F.sigmoid(root_i.gate_low1_param).item(), F.sigmoid(root_i.gate_upp1_param).item(), F.sigmoid(root_i.gate_low2_param).item(), F.sigmoid(root_i.gate_upp2_param).item()]

        if is_old_format:
            leaf_f = results_dict['learned_leaf_gate']
        else:
            leaf_f = results_dict['gates_per_iter'][i][1]
        leaf_gate_f = [F.sigmoid(leaf_f.gate_low1_param).item(), F.sigmoid(leaf_f.gate_upp1_param).item(), F.sigmoid(leaf_f.gate_low2_param).item(), F.sigmoid(leaf_f.gate_upp2_param).item()]

        if is_old_format:
            root_f = results_dict['learned_root_gate']
        else:
            root_f = results_dict['gates_per_iter'][i][0]
        root_gate_f = [F.sigmoid(root_f.gate_low1_param).item(), F.sigmoid(root_f.gate_upp1_param).item(), F.sigmoid(root_f.gate_low2_param).item(), F.sigmoid(root_f.gate_upp2_param).item()]


        #print(leaf_gate_i)
        #print(leaf_gate_f)
        #print(ref_gates[1])
   
        #print(root_gate_i)
        #print(root_gate_f)
        #print(ref_gates[0])


        size = 10
        n_epoch_eval = 20#results_dict['NUM_EPOCHS_PER_EVALUATION'] FIX ME this needs to be a parameter of the funciton call.
        axes = [[pos_axes[0], neg_axes[0]], [pos_axes[1], neg_axes[1]]]
        axes[0][0].scatter(pos_normalized_sample[:, 0], pos_normalized_sample[:, 1], s=size, c='r')
        within_gate_pos = filter_gate(pos_normalized_sample, root_gate_f)
        axes[1][0].scatter(within_gate_pos[:, 2], within_gate_pos[:, 3], s=size, c='r')
        plot_gate(axes[0][0], root_gate_i, 'g', 'init_gate', dashed=True)
        plot_gate(axes[0][0], root_gate_f, 'g', 'final_gate')
        plot_gate(axes[0][0], ref_gates[0], 'b', 'by_inspection')
        axes[0][0].legend()
        if i==0:
            axes[0][0].set_ylabel('Class 1, M1 vs M2')
        axes[0][0].set_title('Iteration %d' %(n_epoch_eval * i))

        plot_gate(axes[1][0], leaf_gate_i, 'g', 'init_gate', dashed=True)
        plot_gate(axes[1][0], leaf_gate_f, 'g', 'learned_gate')
        plot_gate(axes[1][0], ref_gates[1], 'b', 'by_inspection')
        if i==0:
            axes[1][0].set_ylabel('Class 1, M3 vs M4')
        axes[1][0].legend()


        axes[0][1].scatter(neg_normalized_sample[:, 0], neg_normalized_sample[:, 1], s=size)
        within_gate_neg = filter_gate(neg_normalized_sample, root_gate_f)
        axes[1][1].scatter(within_gate_neg[:, 2], within_gate_neg[:, 3], s=size)
        plot_gate(axes[0][1], root_gate_i, 'g', 'init_gate', dashed=True)
        plot_gate(axes[0][1], root_gate_f, 'g', 'final_gate')
        plot_gate(axes[0][1], ref_gates[0], 'b', 'by_inspection')
        if i==0:
            axes[0][1].set_ylabel('Class 2, M1 vs M2')
        axes[0][1].legend()

        plot_gate(axes[1][1], leaf_gate_i, 'g', 'init_gate', dashed=True)
        plot_gate(axes[1][1], leaf_gate_f, 'g', 'learned_gate')
        plot_gate(axes[1][1], ref_gates[1], 'b', 'by_inspection')
        if i==0:
            axes[1][1].set_ylabel('Class 2, M3 vs M4')
        axes[1][1].legend() 
        return axes 
        
def plot_box(axes, x1, x2, y1, y2, color, label, dashed=False, lw=3):
    dash = [3,1]
    if dashed:
        axes.plot([x1, x1], [y1, y2], c=color, label=label, dashes=dash, linewidth=lw)
        axes.plot([x1, x2], [y1, y1], c=color, dashes=dash, linewidth=lw)
        axes.plot([x2, x2], [y1, y2], c=color, dashes=dash, linewidth=lw)
        axes.plot([x2, x1], [y2,y2], c=color, dashes=dash, linewidth=lw)
    else:
        axes.plot([x1, x1], [y1, y2], c=color, label=label, linewidth=lw)
        axes.plot([x1, x2], [y1, y1], c=color, linewidth=lw)
        axes.plot([x2, x2], [y1, y2], c=color, linewidth=lw)
        axes.plot([x2, x1], [y2,y2], c=color, linewidth=lw)
    return axes

def plot_gate(axes, gate, color, label, dashed=False):
    plot_box(axes, gate[0], gate[1], gate[2], gate[3], color, label, dashed=dashed)

    
if __name__== '__main__':
    #with open('../../output/synth/full_batch/batch_size=full_batch_scale=.4_te-tr=1000_100.pkl', 'rb') as f:
    #with open('../../output/synth/full_batch/testing_scale=.1_middle_init.pkl', 'rb') as f:
    with open('../../output/synth/full_batch/testing_synthex1_middle_init.pkl', 'rb') as f:
        results_dict, params_dict = pickle.load(f)
    SYNTH_DATA_DIR = '../../data/synth/'
    DATASET_NAME = 'synthex1.pkl'#'synthex_scale=.1_N=2000.pkl'
    seed = 0
    np.random.seed(seed)
    with open(SYNTH_DATA_DIR + DATASET_NAME, 'rb') as f:
        samples, labels = pickle.load(f)
        indices = np.random.randint(0, len(samples), len(samples))
        samples = [samples[i] for i in indices]
        labels = [labels[i] for i in indices]
        #print(labels)
        normalized_samples, offset, scale = dh.normalize_x_list(samples)

        FEATURE2ID = {'M1':0, 'M2':1, 'M3':2, 'M4':3}
    #print('Total Training Time: ', results_dict['training_time']/60)
    #plot_results(results_dict, params_dict)

    #print('init gates: ', results_dict['root_init_gate'], results_dict['leaf_gate_init']) 
    print( labels[0])
    print(labels[2])
    print(results_dict['accs'])
    #plot_gates(results_dict, params_dict, normalized_samples[0], normalized_samples[2], offset, scale)
    n_epoch_eval = params_dict['NUM_EPOCHS_PER_EVALUATION']
    iters_idxs = [0//n_epoch_eval, 2, 4, 299//n_epoch_eval]
    make_motion_plot(results_dict, normalized_samples[0], normalized_samples[2], iters_idxs, is_old_format=False)
