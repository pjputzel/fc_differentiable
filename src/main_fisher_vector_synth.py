from random import shuffle
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import numpy as np
from utils.bayes_gate_pytorch_sigmoid_trans import *
import utils.load_data as dh
from sklearn.model_selection import train_test_split
import cyvlfeat.fisher.fisher as get_fisher_vectors
import cyvlfeat.gmm.gmm as gmm
from sklearn.svm import LinearSVC
import time
import torch
import pickle
from copy import deepcopy
from utils import plot as util_plot
from sklearn.model_selection import KFold

def run_fisher_4d(samples, labels, test_samples, test_labels, K=10):
    start_time = time.time()    

    #Normalization
    normalize_FV = True


    #output directory for results
    OUT_DIR = '../output/synth/fisher_vector/'
    OUT_NAME = 'scale=.4_' +  'te-tr=1000_%d.pkl' %num_samples

    #Fit GMM to all training data
    means, covar, priors, _, _ = gmm(np.concatenate(samples), n_clusters=K)
    means = means.transpose().astype('float32')
    covar = covar.transpose().astype('float32')
    priors = priors.astype('float32')
    
    #Get Fisher Vectors
    fvs = np.array([get_fisher_vectors(sample.transpose().astype('float32'), means, covar, priors, normalized=normalize_FV) for sample in samples])
    test_fvs = np.array([get_fisher_vectors(test_sample.transpose().astype('float32'), means, covar, priors, normalized=normalize_FV) for test_sample in test_samples])

    #Train linear SVM
    svm = LinearSVC()
    svm.fit(fvs, labels)
    acc = svm.score(test_fvs, test_labels)

    print('Accuracy is:',  acc)
    print('Runtime is %d seconds' %(time.time() - start_time))
    return acc

if __name__ == '__main__':
    K = 100

    SYNTH_DATA_DIR = '../data/synth/'
    DATASET_NAME = 'synthex_scale=.4_N=2000.pkl'
    seed = 1
    np.random.seed(seed)
    num_samples = 100
    if num_samples > 1000:
        raise ValueError('Number of samples must be less than test set size')
    with open(SYNTH_DATA_DIR + DATASET_NAME, 'rb') as f:
        samples, labels = pickle.load(f)
        indices = np.random.randint(0, len(samples), len(samples))
        samples = [samples[i] for i in indices]
        labels = [labels[i] for i in indices]
        labels = np.array(labels)

        synth_test_samples = samples[1000:]
        synth_test_labels = labels[1000:]
        synth_labels = labels[0:num_samples]
        synth_samples = samples[0:num_samples]


    CLL_DATA_DIR = '../data/cll/'
    DATASET_NAME = 'DAFI_gate3-From_Max.pkl'
    np.random.seed(seed)
    num_tr_samples = 90 #rest are for testing
    with open(CLL_DATA_DIR + DATASET_NAME, 'rb') as f:
        samples, labels, ids = pickle.load(f)
        labels = np.array([1 if label=='yes' else 0 for label in labels])
        samples = [sample.astype('float32') for sample in samples]
        indices = np.random.randint(0, len(samples), len(samples))
        samples = [samples[i] for i in indices]
        labels = [labels[i] for i in indices]
        labels = np.array(labels)

        #CLL_test_samples = samples[num_tr_samples:]
        #CLL_test_labels = labels[num_tr_samples:]
        #CLL_labels = labels[0:num_tr_samples]
        #CLL_samples = samples[0:num_tr_samples]

    #print('Running synth experiment')
    #run_fisher_4d(synth_samples, synth_labels, synth_test_samples, synth_test_labels, K=K)

    print('Running CLL experiments')
    num_folds = 10
    folds = KFold(num_folds)
    i = 0
    accs = []
    for tr_idx, te_idx in folds.split(samples):
        tr_samples = [samples[idx] for idx in tr_idx]
        te_samples = [samples[idx] for idx in te_idx]

        tr_labels = labels[tr_idx]
        te_labels = labels[te_idx]
        print('Fold %d'%i)
        i += 1
        acc = run_fisher_4d(tr_samples, tr_labels, te_samples, te_labels, K=K)
        accs.append(acc)

    avg_acc = sum(accs)/num_folds
    print('Average accuracy is %f' %avg_acc)
    
