import matplotlib.pyplot as plt
import matplotlib
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

def filter_outside_gate(sample, gate):
    idxs = (sample[:, 0] > gate[0]) & (sample[:, 0] < gate[1]) & (sample[:, 1] > gate[2]) & (sample[:, 1] < gate[3])
    idxs = ~idxs
    return sample[idxs]
def get_reference_gates(offset, scale):
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

    return ref_gates


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


def get_leaf_root(results_dict, is_old_format):

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
    return leaf_gate_i, leaf_gate_f, root_gate_i, root_gate_f

def plot_single_gate_arrows(results_dict, ref_gates, sample, i, size=10, is_old_format=False, colors=('b', 'r'), sample_str='What class am I?', saveas='name_me.png'):
    matplotlib.use('TKAgg')
    matplotlib.rcParams['font.size'] = 14
    matplotlib.rcParams['font.family'] = 'serif'
    
    leaf_i, leaf_f, root_i, root_f = get_leaf_root(results_dict, is_old_format)

    fig, axes = plt.subplots(2, 1, figsize=(10, 10))
    for ax in axes:
        ax.set(adjustable='box-forced', aspect='equal')

    axes[0].scatter(sample[:, 0], sample[:, 1], s=size)
    axes[0].set_xlim(0, 1)
    axes[0].set_ylim(0, 1)
    
    within_gate = filter_gate(sample, root_f)
   
    plot_gate(axes[0], root_i, colors[1], 'Initial Gate', dashed=True)
    plot_gate(axes[0], root_f, colors[1], 'Final Gate')
    axes[0].legend(loc=1)
    axes[0].set_title('Root Node ' + sample_str)
    axes[0].set_xlabel('M1')
    axes[0].set_ylabel('M2')


    axes[1].scatter(within_gate[:, 2], within_gate[:, 3], s=size)
    plot_gate(axes[1], leaf_i, colors[1], 'Initial Gate', dashed=True)
    plot_gate(axes[1], leaf_f, colors[1], 'Final Gate')
    axes[1].legend(loc=1)
    axes[1].set_title('Leaf Node ' + sample_str)
    axes[1].set_xlabel('M3')
    axes[1].set_ylabel('M4')

    axes[0].annotate('', xy=(.5, -0.5), xycoords='axes fraction', xytext=(.5, -0.17), \
            arrowprops=dict(arrowstyle='simple', color='b'))
    
    plt.subplots_adjust(hspace=.65)
    plt.savefig(saveas, bbox_inches='tight')
    plt.show() 



def plot_single_gate(results_dict, ref_gates, pos_normalized_sample, neg_normalized_sample, i, is_old_format=False, plot_final_gate=True, include_labels=False, leaf_label='Inside Learned Root Gate', plot_within_gate=True, colors=('b', 'g'), include_by_inspection=False):



        matplotlib.use('TKAgg')
        matplotlib.rcParams['font.size'] = 10
        matplotlib.rcParams['font.family'] = 'serif'

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
        fig, axes = plt.subplots(2, 2, figsize=(10, 10))

        #for ax in axes:
        #    ax[0].set(adjustable='box-forced', aspect='equal')
        #    ax[1].set(adjustable='box-forced', aspect='equal')

        axes[0][0].scatter(pos_normalized_sample[:, 0], pos_normalized_sample[:, 1], s=size)
        axes[0][0].set_xlim(0, 1)
        axes[0][0].set_ylim(0, 1)
        if plot_within_gate:
            within_gate_pos = filter_gate(pos_normalized_sample, root_gate_f)
        else:
            within_gate_pos = pos_normalized_sample
        axes[1][0].scatter(within_gate_pos[:, 2], within_gate_pos[:, 3], s=size)
        plot_gate(axes[0][0], root_gate_i, colors[1], 'Initial Gate', dashed=True)
        if plot_final_gate:
            plot_gate(axes[0][0], root_gate_f, colors[1], 'Final Gate')
        if include_by_inspection:
            plot_gate(axes[0][0], ref_s[0], colors[0], 'By Inspection')
        axes[0][0].legend()
        if include_labels:
            axes[0][0].set_title('Positive Root')
            axes[0][0].set_xlabel('M1')
            axes[0][0].set_ylabel('M2')


        plot_gate(axes[1][0], leaf_gate_i, colors[1], 'Initial Gate', dashed=True)
        axes[1][0].set_xlim(0, 1)
        axes[1][0].set_ylim(0, 1)
        if plot_final_gate:
            plot_gate(axes[1][0], leaf_gate_f, colors[1], 'Learned Gate')
        if include_by_inspection:
            plot_gate(axes[1][0], ref_gates[1], colors[0], 'By Inspection')
        axes[1][0].legend()
        if include_labels:
            axes[1][0].set_title(leaf_label)
            axes[1][0].set_xlabel('M3')
            axes[1][0].set_ylabel('M4')


        axes[0][1].scatter(neg_normalized_sample[:, 0], neg_normalized_sample[:, 1], s=size)
        axes[0][1].set_xlim(0, 1)
        axes[0][1].set_ylim(0, 1)
        if plot_within_gate:
            within_gate_neg = filter_gate(neg_normalized_sample, root_gate_f)
        else:
            within_gate_neg = neg_normalized_sample
        axes[1][1].scatter(within_gate_neg[:, 2], within_gate_neg[:, 3], s=size)
        plot_gate(axes[0][1], root_gate_i, colors[1], 'Initial Gate', dashed=True)
        if plot_final_gate:
            plot_gate(axes[0][1], root_gate_f, colors[1], 'Final Gate')
        if include_by_inspection:
            plot_gate(axes[0][1], ref_gates[0], colors[0], 'By Inspection')
        axes[0][1].legend()
        if include_labels:
            axes[0][1].set_title('Negative Root')
            axes[0][1].set_xlabel('M1')
            axes[0][1].set_ylabel('M2')

        plot_gate(axes[1][1], leaf_gate_i, colors[1], 'Initial Gate', dashed=True)
        axes[1][1].set_xlim(0, 1)
        axes[1][1].set_ylim(0, 1)
        if plot_final_gate:
            plot_gate(axes[1][1], leaf_gate_f, colors[1], 'Learned Gate')
        if include_by_inspection:
            plot_gate(axes[1][1], ref_gates[1], colors[0], 'By Inspection')
        axes[1][1].legend()
        if include_labels:
            axes[1][1].set_title(leaf_label)
            axes[1][1].set_xlabel('M3')
            axes[1][1].set_ylabel('M4')


        #mng = plt.get_current_fig_manager()
        #mng.full_screen_toggle()
        plt.show()
        return leaf_gate_i, leaf_gate_f, root_gate_i, root_gate_f

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
        plot_gate(axes[0][0], root_gate_i, 'g', 'Initial Gate', dashed=True)
        plot_gate(axes[0][0], root_gate_f, 'g', 'Final Gate')
        plot_gate(axes[0][0], ref_gates[0], 'b', 'By Inspection')
        axes[0][0].legend()
        if i==0:
            axes[0][0].set_ylabel('Class 1, M1 vs M2')
        axes[0][0].set_title('Iteration %d' %(n_epoch_eval * i))

        plot_gate(axes[1][0], leaf_gate_i, 'g', 'Initial Gate', dashed=True)
        plot_gate(axes[1][0], leaf_gate_f, 'g', 'Learned Gate')
        plot_gate(axes[1][0], ref_gates[1], 'b', 'By Inspection')
        if i==0:
            axes[1][0].set_ylabel('Class 1, M3 vs M4')
        axes[1][0].legend()


        axes[0][1].scatter(neg_normalized_sample[:, 0], neg_normalized_sample[:, 1], s=size)
        within_gate_neg = filter_gate(neg_normalized_sample, root_gate_f)
        axes[1][1].scatter(within_gate_neg[:, 2], within_gate_neg[:, 3], s=size)
        plot_gate(axes[0][1], root_gate_i, 'g', 'Initial Gate', dashed=True)
        plot_gate(axes[0][1], root_gate_f, 'g', 'Final Gate')
        plot_gate(axes[0][1], ref_gates[0], 'b', 'By Inspection')
        if i==0:
            axes[0][1].set_ylabel('Class 2, M1 vs M2')
        axes[0][1].legend()

        plot_gate(axes[1][1], leaf_gate_i, 'g', 'Initial Gate', dashed=True)
        plot_gate(axes[1][1], leaf_gate_f, 'g', 'Learned Gate')
        plot_gate(axes[1][1], ref_gates[1], 'b', 'By Inspection')
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
    print(params_dict)
    #plot_gates(results_dict, params_dict, normalized_samples[0], normalized_samples[2], offset, scale)

    i = -1
    pos_normalized_sample = normalized_samples[0]
    neg_normalized_sample = normalized_samples[2]

    ref_gates = get_reference_gates(offset, scale)
    leaf_gate_i, leaf_gate_f, root_gate_i, root_gate_f =  plot_single_gate(results_dict, ref_gates, pos_normalized_sample, neg_normalized_sample, i, is_old_format=False, plot_final_gate=True, include_labels=True, colors=('k', 'r'))
    outside_gate_pos = filter_outside_gate(pos_normalized_sample, root_gate_f)
    outside_gate_neg = filter_outside_gate(neg_normalized_sample, root_gate_f)
    
    #leaf_gate_i, leaf_gate_f, root_gate_i, root_gate_f =  plot_single_gate(results_dict, ref_gates, outside_gate_pos, outside_gate_neg, i, is_old_format=False, plot_final_gate=True, include_labels=True, leaf_label='Outside Learned Root Gate', plot_within_gate=False, colors=('k', 'r'))

    plot_single_gate_arrows(results_dict, ref_gates, pos_normalized_sample, i, is_old_format=False, sample_str='(Positive Sample)', saveas='positive_gates_synth.png')
    plot_single_gate_arrows(results_dict, ref_gates, neg_normalized_sample, i, is_old_format=False, sample_str='(Negative Sample)', saveas='negative_gates_synth.png')
    plt.figure(figsize=(5, 5))
    plt.gca().set(adjustable='box-forced', aspect='equal')
    plt.scatter(outside_gate_pos[:, 2], outside_gate_pos[:, 3], c='b', s=10)
    plt.xlabel('M3')
    plt.ylabel('M4')
    plt.title('Cells Outside the Learned Root Gate')
    plt.savefig('noise.png', bbox_inches='tight')
    plt.show()
    n_epoch_eval = params_dict['NUM_EPOCHS_PER_EVALUATION']
    #iters_idxs = [0//n_epoch_eval, 2, 4, 299//n_epoch_eval]
    #make_motion_plot(results_dict, normalized_samples[0], normalized_samples[2], iters_idxs, is_old_format=False)

    #plt.plot([0, 20, 40, 60, 100, 120, 140, 160, 180, 200, 220, 240, 260, 280, 300, 320, 340, 360 , 380, 400] , results_dict['accs'])
