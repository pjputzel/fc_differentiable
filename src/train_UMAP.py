import time
import torch
import numpy as np
from utils.MetricsTracker import MetricsTracker
from train import run_train_only_logistic_regression
from utils.SchedulerFactory import create_scheduler

def run_train_model(model, train_params, data_input):
    start = time.time()
    tracker = setup_tracker(model, data_input, train_params['metrics_to_eval'])
    optimizer_gates = init_gate_optimizer(model, train_params['learning_rate_gates'])
    # note that learning rate gates will already be checked for equality with learning rate classifier
    # before this step in the code to make sure the lr setting is explicit in joint case
    full_optimizer = torch.optim.Adam(model.parameters(), train_params['learning_rate_gates'])

    gates_scheduler = create_scheduler(optimizer_gates, train_params)
    full_scheduler = create_scheduler(full_optimizer, train_params)
    if train_params['conv_thresh']:
        if train_params['descent_type'] == 'coordinate_descent':
            run_train_until_convergence_coord(model, train_params, data_input, tracker, optimizer_gates, gates_scheduler, train_params['conv_thresh'])
        elif train_params['descent_type'] == 'joint_descent':
            run_train_until_convergence_joint(model, train_params, data_input, tracker, full_optimizer, scheduler, gates_train_params['conv_thresh'])
        else:
            raise ValueError('Training type not recognized. Options are coordinate_descent and joint_descent.')
    else:
        if train_params['descent_type'] == 'coordinate_descent':
            run_train_fixed_epochs_coord(model, train_params, data_input, tracker, optimizer_gates, gates_scheduler)
        elif train_params['descent_type'] == 'joint_descent':
            run_train_fixed_epochs_joint(model, train_params, data_input, tracker, full_optimizer, full_scheduler)
        else:
            raise ValueError('Training type not recognized. Options are coordinate_descent, and joint_descent.')
    print('Training took %.3f seconds' %(time.time() - start))
    return tracker

def set_current_sharpness(model, epoch, annealing_params):
    init_k = annealing_params['init_sharpness']
    final_k = annealing_params['final_sharpness']
    rate = annealing_params['annealing_increase_rate']
    cur_sharpness = init_k + rate * epoch
    if cur_sharpness > final_k: 
        model.logistic_k = torch.tensor(final_k)
    else:
        model.logistic_k = torch.tensor(cur_sharpness)

def run_train_until_convergence_joint(model, train_params, data_input, tracker, full_optimizer, scheduler, conv_thresh):
        prev_loss = torch.tensor(0)
        cur_loss = torch.tensor(np.inf)
        epoch = 0
        while torch.abs(cur_loss - prev_loss) > train_params['conv_thresh']:
            prev_loss = cur_loss
            cur_loss = step_params_jointly(model, data_input, full_optimizer, scheduler)
            if not (train_params['l1_reg_strength'] == 0):
                cur_loss = cur_loss + model.get_l1_loss(train_params['l1_reg_strength'])

            if epoch % train_params['n_epoch_eval'] == 0:
                tracker.update(epoch)
                print_cur_metrics(tracker)
            epoch += 1
        #update tracker one last time
        if not epoch - 1 == tracker.epochs[-1]:
            tracker.update(epoch)
            print_cur_metrics(tracker)

def run_train_until_convergence_coord(model, train_params, data_input, tracker, optimizer_gates, scheduler, conv_thresh):
        prev_loss = torch.tensor(0)
        cur_loss = torch.tensor(np.inf)
        epoch = 0
        while torch.abs(cur_loss - prev_loss) > train_params['conv_thresh']:
            if train_params['annealing']:
                set_current_sharpness(model, epoch, train_params['annealing'])
    
            prev_loss = cur_loss
            fit_classifier_params(model, data_input,\
                train_params['learning_rate_classifier'], l1_reg_strength=train_params['l1_reg_strength']) 
            cur_loss = step_gate_params(model, data_input, optimizer_gates, scheduler)
            
            if epoch % train_params['n_epoch_eval'] == 0:
                tracker.update(epoch)
                print_cur_metrics(tracker)
                if train_params['annealing']:
                    print('current sharpness: ', model.logistic_k)
            epoch += 1
        if not epoch - 1 == tracker.epochs[-1]:
            tracker.update(epoch)
            print_cur_metrics(tracker)

def run_train_fixed_epochs_joint(model, train_params, data_input, tracker, full_optimizer, scheduler):
    for epoch in range(train_params['n_epoch']):
        cur_loss = step_params_jointly(model, data_input, full_optimizer, scheduler)
        print(cur_loss)
        if not (train_params['l1_reg_strength'] == 0):
            cur_loss = cur_loss + model.get_l1_loss(train_params['l1_reg_strength'])
        print(cur_loss)
        if epoch % train_params['n_epoch_eval'] == 0:
            tracker.update(epoch)
            print_cur_metrics(tracker)
    if not epoch - 1 == tracker.epochs[-1]:
        tracker.update(epoch)
        print_cur_metrics(tracker)

def run_train_fixed_epochs_coord(model, train_params, data_input, tracker, optimizer_gates, scheduler):
    for epoch in range(train_params['n_epoch']):
        fit_classifier_params(model, data_input,\
            train_params['learning_rate_classifier'], l1_reg_strength=train_params['l1_reg_strength']) 
        cur_loss = step_gate_params(model, data_input, optimizer_gates, scheduler)
        
        if epoch % train_params['n_epoch_eval'] == 0:
            tracker.update(epoch)
            print_cur_metrics(tracker)
    if not epoch - 1 == tracker.epochs[-1]:
        tracker.update(epoch)
        print_cur_metrics(tracker)


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

def step_params_jointly(model, data_input, full_optimizer, scheduler):
    full_optimizer.zero_grad()
    x_train, y_train = data_input.x_tr, data_input.y_tr
    output = model(x_train, y_train)
    loss = output['loss']
    loss.backward()
    full_optimizer.step()
    scheduler.step()
    return loss
    
 
def step_gate_params(model, data_input, optimizer_gates, scheduler):
    optimizer_gates.zero_grad()
    x_train, y_train = data_input.x_tr, data_input.y_tr
    output = model(x_train, y_train)
    loss = output['loss']
    loss.backward()
    optimizer_gates.step()
    scheduler.step()
    return loss

def print_cur_metrics(tracker):
    display_str = '[epoch %d]' %tracker.epochs[-1] 
    for metric in tracker.metrics:
        if not (metric == 'tr_feat_diff_reg_per_sample'):
            metric_str = '  %s: %.4f  ' %(metric, tracker.metrics[metric][-1])
            display_str += metric_str
    print(display_str)


 

    
