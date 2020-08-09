import os
import numpy as np
import pickle

def load_single_tr_te_acc(path):
    with open(os.path.join(path, 'tracker.pkl'), 'rb') as f:
        tracker = pickle.load(f)
    print(path)
    acc_tr = tracker.metrics['tr_acc'][-1]
    acc_te = tracker.metrics['te_acc'][-1]
    return acc_tr, acc_te

def compute_summary_statistics_umap_with_reg(output_path):
    accs_tr = []
    accs_te = []
    for dir_name in os.listdir(output_path):
        dir_path = os.path.join(output_path, dir_name)
        if not(os.path.isdir(dir_path)):
            continue
        acc_tr, acc_te = load_single_tr_te_acc(dir_path)
        print(acc_tr, acc_te)
        accs_tr.append(acc_tr)
        accs_te.append(acc_te)

    accs_tr = np.array(accs_tr)
    accs_te = np.array(accs_te)
    mean_tr = np.mean(accs_tr)
    mean_te = np.mean(accs_te)
    std_tr = np.std(accs_tr)
    std_te = np.std(accs_te)
    print('Average train accuracy is %.3f, std is %.3f' %(mean_tr, std_tr))
    print('Average test accuracy is %.3f, std is %.3f' %(mean_te, std_te))
    

# done on laptop, moved output there since I used up all my storage on the servers
def compute_summary_statistics_umap_without_reg(output_path):
    pass

if __name__ == '__main__':
    output_path = '../output/umap_with_feat_diff_cv_experiments'
    compute_summary_statistics_umap_with_reg(output_path)
