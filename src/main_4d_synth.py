from random import shuffle
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import numpy as np
from utils.bayes_gate_pytorch_sigmoid_trans import *
import utils.load_data as dh
from sklearn.model_selection import train_test_split
import time
import torch
import pickle
from copy import deepcopy
from utils import plot as util_plot


if __name__ == '__main__':
    start = time.time()
    SYNTH_DATA_DIR = '../data/synth/'
    DATASET_NAME = 'synthex1.pkl'
    with open(SYNTH_DATA_DIR + DATASET_NAME, 'rb') as f:
        samples, labels = pickle.load(f)
        normalized_samples, offset, scale = dh.normalize_x_list(samples)
        normalized_samples = [torch.tensor(sample).float() for sample in normalized_samples]
        labels = torch.tensor(np.array(labels)).float()
        FEATURE2ID = {'M1':0, 'M2':1, 'M3':2, 'M4':3}

        
    
    #output directory for results
    OUT_DIR = '../output/synth/'
    OUT_NAME = 'testing.pkl'
    
    #Sharpness of the logistic smoothing curve, NOT used by the logistic classifier
    LOGISTIC_K = 100
   
    #Regularization Strengths for each term 
    REGULARIZATION_PENALTY = 0
    EMPTYNESS_PENALTY = 0
    GATE_SIZE_PENALTY = 0
    GATE_SIZE_DEFAULT = 1./4

    #Initialization method for the gates
    DAFI_INIT = False
    if DAFI_INIT:
        INIT_METHOD = "dafi_init"
    else:
        INIT_METHOD = "random_init"

    LOSS_TYPE = 'logistic'  # or 'MSE'
    OPTIMIZER = "ADAM"
    learning_rate_classifier = .05
    learning_rate_gates = .5
    
    batch_size = 100
    n_mini_batch_update_gates = 50
    n_mini_batch = len(normalized_samples) // batch_size

    n_epoch = 500
    n_epoch_eval = 20 

    params_dict = {'lr_classifier':learning_rate_classifier, 'lr_gates':learning_rate_gates, 'LOSS_TYPE':LOSS_TYPE, 'OPTIMIZER':'ADAM', 'batch_size':batch_size, 'n_mini_batch_update_gates':n_mini_batch_update_gates, 'n_epoch':n_epoch, 'REGULARIZATION_PENALTY':REGULARIZATION_PENALTY, 'EMPTYNESS_PENALTY':EMPTYNESS_PENALTY, 'GATE_SIZE_DEFAULT':GATE_SIZE_DEFAULT, 'GATE_SIZE_PENALTY':GATE_SIZE_PENALTY, 'LOGISTIC_K':LOGISTIC_K, 'NUM_EPOCHS_PER_EVALUATION':n_epoch_eval}
    print(params_dict)

    
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
    nested_list_init = \
        [
            [[u'M1', 0.500, 0.500], [u'M2', 0.500, 0.500]],
            [
                [
                    [[u'M3', 0.500, 0.500], [u'M4', 0.500, 0.500]],
                    []
                ]
            ]
        ]
    nested_list = dh.normalize_nested_tree(nested_list, offset, scale, FEATURE2ID)
    nested_list_init = dh.normalize_nested_tree(nested_list_init, offset, scale, FEATURE2ID)
    reference_tree = ReferenceTree(nested_list, FEATURE2ID)
    init_tree = ReferenceTree(nested_list_init, FEATURE2ID)
    if DAFI_INIT:
        init_tree = None

    start = time.time()
    model_tree = ModelTree(reference_tree, logistic_k=LOGISTIC_K, regularisation_penalty=REGULARIZATION_PENALTY,
                           emptyness_penalty=EMPTYNESS_PENALTY, gate_size_penalty=GATE_SIZE_PENALTY,
                           init_tree=init_tree, loss_type=LOSS_TYPE, gate_size_default=GATE_SIZE_DEFAULT)
    keys = [ key for key in model_tree.children_dict.keys()]
    

    
    
    results_dict = {'losses': None, 'log_losses': None, 'reg_size_losses': None, 'ref_reg_losses': None, 'accs':None, 'precs':None, 'recalls':None, 'log_decision_boundaries':None, 'root_init_gate':deepcopy(model_tree.root), 'leaf_gate_init':deepcopy(model_tree.children_dict[str(id(model_tree.root))][0]), 'learned_root_gate':None, 'learned_leaf_gate': None}
    losses = []
    log_losses = []
    size_reg_losses = []
    ref_reg_losses = []

    accs = []
    precs = []
    recalls = []
    
    log_decision_boundaries = []
    
    
    classifier_params = [model_tree.linear.weight, model_tree.linear.bias]
    gates_params = [p for p in model_tree.parameters() if p not in classifier_params]
    if OPTIMIZER == "SGD":
        optimizer_classifier = torch.optim.SGD(classifier_params, lr=learning_rate_classifier)
        optimizer_gates = torch.optim.SGD(gates_params, lr=learning_rate_gates)
    elif OPTIMIZER == 'ADAM':
        optimizer_classifier = torch.optim.Adam(classifier_params, lr=learning_rate_classifier)
        optimizer_gates = torch.optim.Adam(gates_params, lr=learning_rate_gates)
    else:
        raise ValueError('Optimizer type not found')



    for epoch in range(n_epoch):

        # shuffle training data
        idx_shuffle = np.array([i for i in range(len(normalized_samples))])
        shuffle(idx_shuffle)
        normalized_samples = [normalized_samples[_] for _ in idx_shuffle]
        labels = labels[idx_shuffle]

        for i in range(n_mini_batch):
            # generate mini batch data
            idx_batch = [j for j in range(batch_size * i, batch_size * (i + 1))]
            x_batch = [normalized_samples[j] for j in idx_batch]
            y_batch = labels[idx_batch]

            # zero the parameter gradients
            optimizer_gates.zero_grad()
            optimizer_classifier.zero_grad()

            # forward + backward + optimize
            output = model_tree(x_batch, y_batch)
            loss = output['loss']
            loss.backward()
    
            #Step classification optimizer more than gate optimizer
            if (n_mini_batch * epoch + i) % n_mini_batch_update_gates == 0:
                optimizer_gates.step()
            else:
                optimizer_classifier.step()

        #Evaluate every n_epoch_eval iterations
        if epoch % n_epoch_eval == 0:
            print(model_tree)
            log_decision_boundaries.append((torch.exp(-model_tree.linear.bias.detach() / model_tree.linear.weight.detach())))
            output_train = model_tree(normalized_samples, labels)
            labels_pred = (output_train['y_pred'].detach().numpy() > 0.5) * 1.0
            losses.append(output_train['loss'])
            log_losses.append(output_train['log_loss'])
            ref_reg_losses.append(output_train['ref_reg_loss'])
            size_reg_losses.append(output_train['size_reg_loss'])
            accs.append(sum(labels_pred == labels.numpy()) * 1.0 / len(normalized_samples))
            precs.append(precision_score(labels.numpy(), labels_pred, average='macro'))
            recalls.append(recall_score(labels.numpy(), labels_pred, average='macro'))
            print('Epoch: ', epoch)
            print('Current decision boundary:', log_decision_boundaries[-1])
            print('Current Accuracy:', accs[-1])
            print('Total Loss, Logistic Loss, Ref Reg Loss, Size Reg Loss: [%.3f, %.3f, %.4f, %.4f]' %(losses[-1], log_losses[-1], ref_reg_losses[-1], size_reg_losses[-1]))
    
    #save results
    results_dict['losses'] = losses
    results_dict['log_losses'] = log_losses
    results_dict['ref_reg_losses'] = ref_reg_losses
    results_dict['size_reg_losses'] = size_reg_losses
    results_dict['accs'] = accs
    results_dict['precs'] = precs
    results_dict['recalls'] = recalls
    results_dict['log_decision_boundaries'] = log_decision_boundaries

    results_dict['learned_leaf_gate'] = deepcopy(model_tree.children_dict[str(id(model_tree.root))][0])
    results_dict[ 'learned_root_gate'] = model_tree.root
    print(results_dict)
 
    with open(OUT_DIR + OUT_NAME, 'wb') as f:
        pickle.dump((results_dict, params_dict), f) 
     
    
     
