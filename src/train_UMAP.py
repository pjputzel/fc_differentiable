import time
import torch
import numpy as np
from utils.MetricsTracker import MetricsTracker
from train import run_train_only_logistic_regression

def run_train_model(model, train_params, data_input):
    start = time.time()
    tracker = setup_tracker(model, data_input, train_params['metrics_to_eval'])
    optimizer_gates = init_gate_optimizer(model, train_params['learning_rate_gates'])
    # note that learning rate gates will already be checked for equality with learning rate classifier
    # before this step in the code to make sure the lr setting is explicit in joint case
    full_optimizer = torch.optim.Adam(model.parameters(), train_params['learning_rate_gates'])
    if train_params['conv_thresh']:
        if train_params['descent_type'] == 'coordinate_descent':
            run_train_until_convergence_coord(model, train_params, data_input, tracker, optimizer_gates, train_params['conv_thresh'])
        elif train_params['descent_type'] == 'joint_descent':
            run_train_until_convergence_joint(model, train_params, data_input, tracker, full_optimizer, train_params['conv_thresh'])
        else:
            raise ValueError('Training type not recognized. Options are coordinate_descent and joint_descent.')
    else:
        if train_params['descent_type'] == 'coordinate_descent':
            run_train_fixed_epochs_coord(model, train_params, data_input, tracker, optimizer_gates)
        elif train_params['descent_type'] == 'joint_descent':
            run_train_fixed_epochs_joint(model, train_params, data_input, tracker, full_optimizer)
        else:
            raise ValueError('Training type not recognized. Options are coordinate_descent, and joint_descent.')
    print('Training took %.3f seconds' %(time.time() - start))
    return tracker

def run_train_until_convergence_joint(model, train_params, data_input, tracker, full_optimizer, conv_thresh):
        prev_loss = torch.tensor(0)
        cur_loss = torch.tensor(np.inf)
        epoch = 0
        while torch.abs(cur_loss - prev_loss) > train_params['conv_thresh']:
            prev_loss = cur_loss
            cur_loss = step_params_jointly(model, data_input, full_optimizer)
            if epoch % train_params['n_epoch_eval'] == 0:
                tracker.update(epoch)
                print_cur_metrics(tracker)
            epoch += 1
        #update tracker one last time
        if not epoch - 1 == tracker.epochs[-1]:
            tracker.update(epoch)


def run_train_until_convergence_coord(model, train_params, data_input, tracker, optimizer_gates, conv_thresh):
        prev_loss = torch.tensor(0)
        cur_loss = torch.tensor(np.inf)
        epoch = 0
        while torch.abs(cur_loss - prev_loss) > train_params['conv_thresh']:
            prev_loss = cur_loss
            fit_classifier_params(model, data_input,\
                train_params['learning_rate_classifier'], l1_reg_strength=train_params['l1_reg_strength']) 
            cur_loss = step_gate_params(model, data_input, optimizer_gates)
            
            if epoch % train_params['n_epoch_eval'] == 0:
                tracker.update(epoch)
                print_cur_metrics(tracker)
            epoch += 1
        if not epoch - 1 == tracker.epochs[-1]:
            tracker.update(epoch)

def run_train_fixed_epochs_joint(model, train_params, data_input, tracker, full_optimizer):
    for epoch in range(train_params['n_epoch']):
        cur_loss = step_params_jointly(model, data_input, full_optimizer)
        if epoch % train_params['n_epoch_eval'] == 0:
            tracker.update(epoch)
            print_cur_metrics(tracker)
    if not epoch - 1 == tracker.epochs[-1]:
        tracker.update(epoch)

def run_train_fixed_epochs_coord(model, train_params, data_input, tracker, optimizer_gates):
    for epoch in range(train_params['n_epoch']):
        fit_classifier_params(model, data_input,\
            train_params['learning_rate_classifier'], l1_reg_strength=train_params['l1_reg_strength']) 
        cur_loss = step_gate_params(model, data_input, optimizer_gates)
        
        if epoch % train_params['n_epoch_eval'] == 0:
            tracker.update(epoch)
            print_cur_metrics(tracker)
    if not epoch - 1 == tracker.epochs[-1]:
        tracker.update(epoch)


def setup_tracker(model, data_input, metrics):
    tracker = MetricsTracker(model, data_input, metrics) 
    tracker.update(0)
    return tracker


def init_gate_optimizer(model, lr_gates):
    classifier_params = [model.linear.weight, model.linear.bias]
    gates_params = [
                    p for p in model.parameters() if not 
                    any(p is d_ for d_ in classifier_params)
                   ]
    optimizer_gates = torch.optim.Adam(gates_params, lr=lr_gates)
    return optimizer_gates
    
def fit_classifier_params(model, data_input, lr, l1_reg_strength=0):
   run_train_only_logistic_regression(model, data_input.x_tr,\
        data_input.y_tr, lr, verbose=False, l1_reg_strength=l1_reg_strength)

def step_params_jointly(model, data_input, full_optimizer):
    full_optimizer.zero_grad()
    x_train, y_train = data_input.x_tr, data_input.y_tr
    output = model(x_train, y_train)
    loss = output['loss']
    loss.backward()
    full_optimizer.step()
    return loss
    
 
def step_gate_params(model, data_input, optimizer_gates):
    optimizer_gates.zero_grad()
    x_train, y_train = data_input.x_tr, data_input.y_tr
    output = model(x_train, y_train)
    loss = output['loss']
    loss.backward()
    optimizer_gates.step()
    return loss

def print_cur_metrics(tracker):
    display_str = '[epoch %d]' %tracker.epochs[-1] 
    for metric in tracker.metrics:
        if not (metric == 'tr_feat_diff_reg_per_sample'):
            metric_str = '  %s: %.4f  ' %(metric, tracker.metrics[metric][-1])
            display_str += metric_str
    print(display_str)


 

    
