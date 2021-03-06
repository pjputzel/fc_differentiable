import pickle
import numpy as np
import matplotlib.pyplot as plt
import os

SIZES = [0.025, 0.05, .1, .2, .3]

def simple_diagnostics_plot(trackers_path, trackers_path2=None):
    with open(trackers_path, 'rb') as f:
        trackers = pickle.load(f)
    tr_losses, tr_accs = get_results_per_converged_gate_tr(trackers)
    te_losses, te_accs = get_results_per_converged_gate_te(trackers)
    x_ticks = np.arange(len(tr_losses)) + 1
    plt.plot(x_ticks, tr_losses, label='tr log loss')
    plt.plot(x_ticks, tr_accs, label='tr acc')
    plt.legend()
    plt.savefig('acc_and_loss_as_function_of_number_of_gates_plot_tr.png')
    plt.clf()

    x_ticks = np.arange(len(te_losses)) + 1
    plt.plot(x_ticks, te_losses, label='te log loss')
    plt.plot(x_ticks, te_accs, label='te acc')
    plt.savefig('acc_and_loss_as_function_of_number_of_gates_plot_te.png')
    plt.legend()
    plt.clf()

    update_last_tracker_to_have_features_at_convergence(trackers)   
    pos_features_tr = np.array([np.log10(np.exp(feature)) for i, feature in enumerate(trackers[-1].metrics['tr_features'][0][:, 0].detach().numpy()) if bool(trackers[-1].data_input.y_tr[i] == 1)])[:, np.newaxis]
    neg_features_tr = np.array([np.log10(np.exp(feature)) for i, feature in enumerate(trackers[-1].metrics['tr_features'][0][:, 0].detach().numpy()) if bool(trackers[-1].data_input.y_tr[i] == 0)])[:, np.newaxis]
    print(pos_features_tr.shape, neg_features_tr.shape)

    if trackers_path2:
        with open(trackers_path2, 'rb') as f:
            trackers2 = pickle.load(f)
        update_last_tracker_to_have_features_at_convergence(trackers2)   
        pos_features_tr2 = np.array([np.log10(np.exp(feature)) for i, feature in enumerate(trackers2[-1].metrics['tr_features'][0][:, 0].detach().numpy()) if bool(trackers[-1].data_input.y_tr[i] == 1)])[:, np.newaxis]
        neg_features_tr2 = np.array([np.log10(np.exp(feature)) for i, feature in enumerate(trackers2[-1].metrics['tr_features'][0][:, 0].detach().numpy()) if bool(trackers[-1].data_input.y_tr[i] == 0)])[:, np.newaxis]
        plt.boxplot([pos_features_tr, neg_features_tr, pos_features_tr2, neg_features_tr2], labels = ['pos feats with reg', 'neg feats with reg', 'pos feats without reg', 'neg feats without reg'], showfliers=True)
        plt.savefig('box_plot_with_and_without_feat_diff_reg.png')
        plt.clf()
            
    #neg_features_tr = np.array([tracker.metrics['tr_avg_neg_feat'][-1] for tracker in trackers])[:, np.newaxis]

    pos_features_te = np.array([np.log10(np.exp(feature)) for i, feature in enumerate(trackers[-1].metrics['te_features'][0][:, 0].detach().numpy()) if bool(trackers[-1].data_input.y_te[i] == 1)])[:, np.newaxis]
    neg_features_te = np.array([np.log10(np.exp(feature)) for i, feature in enumerate(trackers[-1].metrics['te_features'][0][:, 0].detach().numpy()) if bool(trackers[-1].data_input.y_tr[i] == 0)])[:, np.newaxis]
    print(pos_features_tr.shape, neg_features_tr.shape)
    #neg_features_tr = np.array([tracker.metrics['tr_avg_neg_feat'][-1] for tracker in trackers])[:, np.newaxis]
    plt.boxplot([np.log(np.power(10, pos_features_tr)), np.log(np.power(10, neg_features_tr))], showfliers=False, labels=['Positive Log-Features Train', 'Negative Log-Features Train'])
    plt.savefig('feat_boxplot_gate_one_tr.png')

    plt.clf()
    plt.boxplot([pos_features_te, neg_features_te], showfliers=False, labels=['Positive Log-Features Test', 'Negative Log-Features Test'])
    plt.savefig('feat_boxplot_gate_one_te.png')
    plt.clf()

    pos_features_tr2 = np.array([feature for i, feature in enumerate(trackers[-1].metrics['tr_features'][0][:, 1].detach().numpy()) if bool(trackers[-1].data_input.y_tr[i] == 1)])[:, np.newaxis]
    neg_features_tr2 = np.array([feature for i, feature in enumerate(trackers[-1].metrics['tr_features'][0][:, 1].detach().numpy()) if bool(trackers[-1].data_input.y_tr[i] == 0)])[:, np.newaxis]
    #neg_features_tr = np.array([tracker.metrics['tr_avg_neg_feat'][-1] for tracker in trackers])[:, np.newaxis]

    pos_features_te2 = np.array([feature for i, feature in enumerate(trackers[-1].metrics['te_features'][0][:, 1].detach().numpy()) if bool(trackers[-1].data_input.y_te[i] == 1)])[:, np.newaxis]
    neg_features_te2 = np.array([feature for i, feature in enumerate(trackers[-1].metrics['te_features'][0][:, 1].detach().numpy()) if bool(trackers[-1].data_input.y_tr[i] == 0)])[:, np.newaxis]

    plt.boxplot([pos_features_tr2, neg_features_tr2], showfliers=False, labels=['Positive Log-Features Train', 'Negative Log-Features Train'])
    plt.savefig('feat_boxplot_gate_two_tr.png')
    plt.clf()

    plt.boxplot([pos_features_te2, neg_features_te2], showfliers=False, labels=['Positive Log-Features Test', 'Negative Log-Features Test'])
    plt.savefig('feat_boxplot_gate_two_te.png')
    plt.clf()


def update_last_tracker_to_have_features_at_convergence(trackers):
    last_tracker = trackers[-1]
    last_tracker.metric_names.append('tr_features')
    last_tracker.metric_names.append('te_features')
    last_tracker.metrics['tr_features'] = []
    last_tracker.metrics['te_features'] = []

    last_tracker.metric_funcs['tr_features'] = last_tracker.compute_tr_features
    last_tracker.metric_funcs['te_features'] = last_tracker.compute_te_features    
    last_tracker.update(-1)

    
def plot_size_vs_accuracy_tr_and_te(trackers_per_size):
    tr_losses, tr_accs = get_results_per_converged_gate_tr(trackers_per_size)
    te_losses, te_accs = get_results_per_converged_gate_te(trackers_per_size)
    plt.xlabel('Box Size')
    plt.ylabel('Accuracy')
    plt.plot(SIZES, tr_accs, label='tr acc')
    plt.plot(SIZES, te_accs, label='te acc')
    plt.legend()
    plt.savefig('accuracy_vs_size.png')
    
    
def plot_size_vs_loss_tr_and_te(trackers_per_run):

    tr_losses, tr_accs = get_results_per_converged_gate_run_avg_tr_size(trackers_per_run)
    te_losses, te_accs = get_results_per_converged_gate_run_avg_te_size(trackers_per_size)
    plt.xlabel('Box Size')
    plt.ylabel('Log Loss')
    plt.plot(SIZES, tr_losses, label='tr log loss')
    plt.plot(SIZES, te_losses, label='te log loss')
    plt.legend()
    plt.savefig('loss_vs_size.png')


def plot_size_vs_tr_avg_features(trackers_per_size):
    tr_avg_pos_feats, tr_avg_neg_feats = get_feats_per_converged_gate_size(trackers_per_size, 'tr')
    plt.xlabel('Box Size')
    plt.ylabel('Log Feature')
    plt.plot(SIZES, tr_avg_pos_feats, label='tr log pos features')
    plt.plot(SIZES, tr_avg_neg_feats, label='tr log neg features')
    plt.legend()
    plt.savefig('avg_feats_vs_size_tr.png')

def plot_size_vs_te_avg_features(trackers_per_size):
    te_avg_pos_feats, te_avg_neg_feats = get_feats_per_converged_gate_size(trackers_per_size, 'te')
    plt.xlabel('Box Size')
    plt.ylabel('Log Feature')
    plt.plot(SIZES, tr_avg_pos_feats, label='te log pos features')
    plt.plot(SIZES, tr_avg_neg_feats, label='te log neg features')
    plt.legend()
    plt.savefig('avg_feats_vs_size_te.png')


def get_feats_per_converged_gate_size(trackers_per_run, split):
    pos_avg_feats = []
    neg_avg_feats = []
    for key in trackers_per_run:
        

        for tracker in trackers_per_size:
            pos_avg_feat, neg_avg_feat = get_avg_feats_one_converged_gate(tracker, split)
            pos_avg_feats.append(pos_avg_feat)
            neg_avg_feats.append(neg_avg_feat)
    return pos_avg_feats, neg_avg_feats
            

def get_avg_results_size_eval(trackers_per_run):
    results = np.zeros([len(trackers_per_run), len(trackers_per_run[0])])
    pos_feats_tr = [avg_results_size_trackers(trackers_per_run[trackers_size_key])['pos_feats_tr'] for trackers_size_key in trackers_per_run]
    neg_feats_tr = [avg_results_size_trackers(trackers_per_run[trackers_size_key])['neg_feats_tr'] for trackers_size_key in trackers_per_run]
    neg_feats_te = [avg_results_size_trackers(trackers_per_run[trackers_size_key])['neg_feats_te'] for trackers_size_key in trackers_per_run]
    pos_feats_te = [avg_results_size_trackers(trackers_per_run[trackers_size_key])['pos_feats_te'] for trackers_size_key in trackers_per_run]
    
    acc_tr = [avg_results_size_trackers(trackers_per_run[trackers_size_key])['avg_acc_tr'] for trackers_size_key in trackers_per_run]
    acc_te = [avg_results_size_trackers(trackers_per_run[trackers_size_key])['avg_acc_te'] for trackers_size_key in trackers_per_run]

    return pos_feats_tr, neg_feats_tr, acc_tr, pos_feats_te, neg_feats_te, acc_te 
def avg_results_size_trackers(trackers_per_run):
    results = {}

    avg_pos_feats_tr = [trackers_per_run[tracker_key].metrics['tr_avg_pos_feat'][-1] for tracker_key in trackers_per_run]
    avg_pos_feats_te = [trackers_per_run[tracker_key].metrics['te_avg_pos_feat'][-1] for tracker_key in trackers_per_run]
    results['pos_feats_tr'] = np.mean(avg_pos_feats_tr)
    results['pos_feats_te'] = np.mean(avg_pos_feats_te)

    avg_neg_feats_tr = [trackers_per_run[tracker_key].metrics['tr_avg_neg_feat'][-1] for tracker_key in trackers_per_run]
    avg_neg_feats_te = [trackers_per_run[tracker_key].metrics['te_avg_neg_feat'][-1] for tracker_key in trackers_per_run]
    results['neg_feats_tr'] = np.mean(avg_neg_feats_tr)
    results['neg_feats_te'] = np.mean(avg_neg_feats_te)

    avg_acc_tr = [trackers_per_run[tracker_key].metrics['tr_acc'][-1] for tracker_key in trackers_per_run]
    results['avg_acc_tr'] = np.mean(avg_acc_tr)

    avg_acc_te = [trackers_per_run[tracker_key].metrics['te_acc'][-1] for tracker_key in trackers_per_run]
    results['avg_acc_te'] = np.mean(avg_acc_te)

    
    return results

def get_results_per_converged_gate_tr(trackers):
    tr_losses = []
    tr_accs = []
    for tracker in trackers:
        tr_loss, tr_acc = get_results_one_converged_gate(tracker)
        tr_losses.append(tr_loss)
        tr_accs.append(tr_acc)

    return tr_losses, tr_accs

def get_results_per_converged_gate_te(trackers):
    te_losses = []
    te_accs = []
    for tracker in trackers:
        te_loss, te_acc = get_results_one_converged_gate_te(tracker)
        te_losses.append(te_loss)
        te_accs.append(te_acc)

    return te_losses, te_accs

def get_results_one_converged_gate_te(tracker):
    tr_loss = float(tracker.metrics['te_log_loss'][-1])
    tr_acc = float(tracker.metrics['tr_acc'][-1])
    return tr_loss, tr_acc

def get_results_one_converged_gate(tracker):
    tr_loss = float(tracker.metrics['tr_log_loss'][-1])
    tr_acc = float(tracker.metrics['tr_acc'][-1])
    return tr_loss, tr_acc

def get_avg_feats_one_converged_gate(tracker, split):
    metric_prefix = split + '_'
    pos_metric_name = metric_prefix + 'avg_pos_feat'
    neg_metric_name = metric_prefic + 'avg_neg_feat'
    avg_pos_feat = tracker.metrics[pos_metric_name][-1]
    avg_neg_feat = tracker.metrics[neg_metric_name][-1]
    return float(avg_pos_feat), float(avg_neg_feat)

def plot_feat_diff_reg_per_iteration_single_tracker(tracker):
    feat_diff_results = tracker.metrics['tr_reg_feat_diff']
    plt.plot(tracker.epochs, feat_diff_results)
    plt.savefig('feat_diff_plot.png')
    plt.clf()

def plot_feat_diff_reg_per_sample(tracker):
    feat_diff_results = tracker.metrics['tr_feat_diff_reg_per_sample'][-1]
    results_to_plot = [result[0] for result in feat_diff_results]
    print(results_to_plot)
    plt.plot(np.arange(len(results_to_plot)), results_to_plot)
    plt.savefig('feat_diff_per_sample.png')
    plt.clf()

def plot_old_model_features_next_to_current_ones(cur_tracker, cur_tracker_reg, old_feats_path):
    old_feats_tr = np.loadtxt(old_feats_path, delimiter=',')[0:34] #first 34 are dev
    cur_feats_tr = cur_tracker.metrics['tr_features'][-1][:, 0].detach().numpy()
    cur_feats_tr_reg = cur_tracker_reg.metrics['tr_features'][-1][:, 0].detach().numpy()

    pos_old_feats = np.log10(old_feats_tr[old_feats_tr[:, -1] == 1])[:, 0]
    pos_cur_feats_tr = [np.log10(np.exp(cur_feat)) for i, cur_feat in enumerate(cur_feats_tr) if bool(cur_tracker.data_input.y_tr[i] == 1)]
    pos_cur_feats_tr_reg = [np.log10(np.exp(cur_feat)) for i, cur_feat in enumerate(cur_feats_tr_reg) if bool(cur_tracker_reg.data_input.y_tr[i] == 1)]

    neg_old_feats = np.log10(old_feats_tr[old_feats_tr[:, -1] == 0])[:, 0]
    print(neg_old_feats)
    neg_cur_feats_tr = [np.log10(np.exp(cur_feat)) for i, cur_feat in enumerate(cur_feats_tr) if bool(cur_tracker.data_input.y_tr[i] == 0)]
    neg_cur_feats_tr_reg = [np.log10(np.exp(cur_feat)) for i, cur_feat in enumerate(cur_feats_tr_reg) if bool(cur_tracker_reg.data_input.y_tr[i] == 0)]

    #add with reg here as well
    plt.boxplot([pos_cur_feats_tr, neg_cur_feats_tr, pos_cur_feats_tr_reg, neg_cur_feats_tr_reg, pos_old_feats, neg_old_feats], labels=['+ feats cur', '- feats cur', '+ feats cur reg', '- feats cur reg', '+ feats old', '- feats old'], showfliers=False)
    plt.savefig('cur_feats_with_and_without_reg_vs_old_results.png')
    plt.clf()



if __name__ == '__main__':
    trackers_dir2 = '../output/one_by_one_clustering_with_loss_heuristic_final_version/trackers.pkl' 
    trackers_dir = '../output/umap_with_feat_diff/trackers.pkl'
    with open(trackers_dir, 'rb') as f:
        trackers = pickle.load(f)
    with open(trackers_dir2, 'rb') as f:
        trackers2 = pickle.load(f)
    update_last_tracker_to_have_features_at_convergence(trackers)
    update_last_tracker_to_have_features_at_convergence(trackers2)


    plot_feat_diff_reg_per_iteration_single_tracker(trackers[-1])
    #trackers_dir = '../output/default_umap/trackers.pkl'
    #trackers_dir = '../output/testing_using_labels_with_transform/trackers.pkl'
    simple_diagnostics_plot(trackers_dir, trackers_path2=trackers_dir2)
    plot_feat_diff_reg_per_sample(trackers[-1])    

    plot_old_model_features_next_to_current_ones(trackers2[-1], trackers[-1], './all_feats_from_paper.csv')
    #trackers_per_run_dir = '../output/eval_grid_final_run/trackers_per_run.pkl'
    #with open(trackers_per_run_dir, 'rb') as f:
    #    trackers_per_run = pickle.load(f)
    #print(get_avg_results_size_eval(trackers_per_run)) 
