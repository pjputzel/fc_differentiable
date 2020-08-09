import matplotlib.pyplot as plt
import pickle
from utils.TransformParameterParser import TransformParameterParser
from utils.DataInput import DataInput
from utils.DepthOneModel import DepthOneModel
from train_UMAP import *
from utils.GateInitializerClustering import GateInitializer

# analysis of the loss surface near the local optimum of a single run using mheur

path_to_saved_model = '../output/repeated_init_testing_grid/model.pkl'
path_to_config = '../configs/umap_default.yaml'
transformer_path = '../output/repeated_init_testing_grid/transformer.pkl'

def main_varying_each_dim_independently(path_to_config, path_to_saved_model):
    params = TransformParameterParser(path_to_config).parse_params()
    set_random_seeds(params)
    model_gates = load_model_gates(path_to_saved_model, params)
    data_input = init_data_input(params, transformer_path)
    make_all_plots(model_gates, params, data_input)

def main_interpolating_gates(path_to_config, path_to_saved_model):
    params = TransformParameterParser(path_to_config).parse_params()
    set_random_seeds(params)
    model_gates = load_model_gates(path_to_saved_model, params)
    data_input = init_data_input(params, transformer_path)
    make_all_plots_interpolated(model_gates, 0, params, data_input) 

def set_random_seeds(params):
    torch.manual_seed(params['random_seed'])
    np.random.seed(params['random_seed'])

def load_model_gates(model_path, params):
    state_dict = torch.load(model_path)
    model_params = params['model_params']
    fake_init_gates = GateInitializer.get_fake_init_gates(params['gate_init_multi_heuristic_params']['num_gates'])
    model = DepthOneModel(fake_init_gates, model_params)
    model.load_state_dict(state_dict)
    return model.get_gates()

def init_data_input(params, transformer_path):
    data_input = DataInput(params['data_params'])
    data_input.split_data()
    with open(transformer_path, 'rb') as f:
        data_transformer = pickle.load(f)
    print(data_transformer)
    data_input.embed_data(data_transformer, \
        params['transform_params']['cells_to_subsample'], 
        params['transform_params']['num_cells_for_transformer']
    )
    data_input.normalize_data()
    data_input.prepare_data_for_training() 
    return data_input

# Interpolates (linearly) between the center of a learned
# converged gate (make sure to use the first learned gate if
# dealing with repeated inits, or just do a run with one single gate) and 
#25% past the boundaries of the data
# range ie 1.25 in each dimension. Also do the run
# without regularization to avoid confusion in intepreting
def make_all_plots_interpolated(model_gates, gate_idx, params, data_input, n_steps=150, figsize=6, percent_beyond_range=.25):
    gate = model_gates[gate_idx]
    print('init gate', gate)
    fig, axes = plt.subplots(2, 1, figsize =(figsize, figsize * 2))
    center = [(gate[1] + gate[0])/2, (gate[3] + gate[2])/2]
    step_x_low, step_y_low = (-percent_beyond_range - center[0])/n_steps, (-percent_beyond_range - center[1])/n_steps
    step_x_high, step_y_high = (1 + percent_beyond_range - center[0])/n_steps, (1 + percent_beyond_range - center[1])/n_steps
    derivative_magnitudes = []
    losses = []
    feats = []
    cur_gate = [center[0], center[0], center[1], center[1]]
    for step in range(n_steps):
        cur_gate = [cur_gate[0] + step_x_low, cur_gate[1] + step_x_high, cur_gate[2] + step_y_low, cur_gate[3] + step_y_high]
        print('is cur gate changins?:', cur_gate)
        cur_model = get_model_with_new_gate(gate_idx, cur_gate, params, model_gates, data_input)
        derivative_magnitude, loss, feat = compute_derivative_magnitude_loss_and_feature(cur_model, data_input, gate_idx)
        losses.append(loss)
        feats.append(feat)
        derivative_magnitudes.append(derivative_magnitude)
    steps_for_plot = np.arange(n_steps)
    axes[0].plot(steps_for_plot, derivative_magnitudes)
    axes[1].plot(steps_for_plot, losses)
    fig.savefig('losses_and_derivatives.png')

def compute_derivative_magnitude_loss_and_feature(cur_model, data_input, gate_idx):
    cur_model.zero_grad()
    output = cur_model(data_input.x_tr, data_input.y_tr)
    loss = output['loss']
    loss.backward()
    derivative_magnitude = get_derivative_magnitude(cur_model.nodes[gate_idx])
    derivative_magnitude = get_derivative_magnitude(cur_model.nodes[1])
    return derivative_magnitude, loss, output['leaf_logp'].cpu().detach().numpy()
    
    

def get_derivative_magnitude(node):
    print(node)
    print(node.side_length_param.grad, node.center1_param.grad, node.center2_param.grad)
    magnitude = node.side_length_param.grad ** 2 + node.center1_param.grad ** 2 + node.center2_param.grad ** 2
    return (magnitude * (1/2)).detach().numpy() 
        
        

# redo without needing model at all, just using saved gates, re-init model with gate locations and logistic params set appropiately for each new cut location
def make_all_plots(model_gates, params, data_input):
    # 2 plots per cut (which there are another 2) dimension, and a row for each gate.
    fig, axes = plt.subplots(len(model_gates), 4, figsize=(len(model_gates) * 4, len(model_gates)))
    for gate_idx, gate in enumerate(model_gates):
        for cut_idx in range(4):
            cut_grid = get_cut_grid(gate, cut_idx)
            losses = compute_loss_over_1d_grid(data_input, gate_idx, cut_idx, params, model_gates)
            axes[gate_idx][cut_idx].plot(cut_grid, losses)
    fig.tight_layout()
    fig.savefig('varying_cuts_to_final_solution.png')

def compute_loss_over_1d_grid(data_input, gate_idx, cut_idx, params, model_gates):
    losses = []
    gate = model_gates[gate_idx]
    cut_grid = get_cut_grid(gate, cut_idx)
    for new_cut in cut_grid:
        new_gate = model_gates[gate_idx]
        new_gate[cut_idx] = new_cut
        new_gate = [['D1', new_gate[0], new_gate[1]], ['D2', new_gate[2], new_gate[3]]]
        model = get_model_with_new_gate(gate_idx, new_gate, params, model_gates, data_input)
        loss = model(data_input.x_tr, data_input.y_tr)['log_loss']
        losses.append(loss)
    return losses

def get_cut_grid(gate, cut_idx, num_points=50):
    if cut_idx == 0 or cut_idx == 2:
        return np.linspace(0, gate[cut_idx], num_points)    
    if cut_idx == 1 or cut_idx == 3:
        return np.linspace(gate[cut_idx], 1, num_points)

def get_tree_cut_idxs(cut_idx):
    if cut_idx == 0:
        return [0, 1]
    elif cut_idx == 1:
        return [0, 2]
    elif cut_idx == 2:
        return [1, 1]
    elif cut_idx == 3:
        return [1, 2]

def get_model_with_new_gate(gate_idx, new_gate, params, model_gates, data_input):
    gates = model_gates
    if type(new_gate[0]) == float:
        new_gate = [['D1', new_gate[0], new_gate[1]], ['D2', new_gate[2], new_gate[3]]]
    formatted_gates = []
    for gate in gates:
        formatted_gates.append([
            ['D1', gate[0], gate[1]],
            ['D2', gate[2], gate[3]]
        ])
    formatted_gates[gate_idx] = new_gate
    train_params = params['train_params']
    model = DepthOneModel(formatted_gates, params['model_params'])
    fit_classifier_params(model, data_input,\
        train_params['learning_rate_classifier'],
        l1_reg_strength=train_params['l1_reg_strength'])
    return model


    
if __name__ == '__main__':
    main_interpolating_gates(path_to_config, path_to_saved_model)
