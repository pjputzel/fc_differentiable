import time
import torch
from utils.MetricsTracker import MetricsTracker
from train import run_train_only_logistic_regression

def run_train_model(model, train_params, data_input):
    start = time.time()
    tracker = setup_tracker(model, data_input, train_params['metrics_to_eval'])
    optimizer_gates = init_gate_optimizer(model, train_params['learning_rate_gates'])
    for epoch in range(train_params['n_epoch']):

        fit_classifier_params(model, data_input,\
            train_params['learning_rate_classifier'], l1_reg_strength=train_params['l1_reg_strength']) 
        step_gate_params(model, data_input, optimizer_gates)
        
        if epoch % train_params['n_epoch_eval'] == 0:
            tracker.update(epoch)
            print_cur_metrics(tracker)
    print('Training took %.3f seconds' %(time.time() - start))
    return tracker

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
 
def step_gate_params(model, data_input, optimizer_gates):
        optimizer_gates.zero_grad()
        x_train, y_train = data_input.x_tr, data_input.y_tr
        output = model(x_train, y_train)
        loss = output['loss']
        loss.backward()
        optimizer_gates.step()

def print_cur_metrics(tracker):
    display_str = '[epoch %d]' %tracker.epochs[-1] 
    for metric in tracker.metrics:
        metric_str = '  %s: %.4f  ' %(metric, tracker.metrics[metric][-1])
        display_str += metric_str
    print(display_str)


 

    
