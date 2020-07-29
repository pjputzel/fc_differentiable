
from sklearn.neighbors import KernelDensity
import copy
import umap
import warnings
import pickle
import numpy as np
import matplotlib.pyplot as plt
from train_UMAP import fit_classifier_params
from utils.TransformParameterParser import TransformParameterParser
from utils.DataInput import DataInput
from utils.GateInitializerPrimKDE import GateInitializerPrimKDE
from utils.GateInitializerClustering import GateInitializerClustering
from utils.DepthOneModel import DepthOneModel
from utils.DataAndGatesPlotter import DataAndGatesPlotterDepthOne
from utils.DataTransformerFactory import DataTransformerFactory
from train_UMAP import run_train_model
from sklearn.cluster import KMeans
import torch
import os
import time
from copy import deepcopy
import utils.utils_load_data as dh
import seaborn as sb


def main_clustering_exapnsion(path_to_params):
    # change to 200 after debugging
    n_clusters_for_expansion = 5
    sample_idxs_to_plot = [0, 1, 2, 3, 4]

    data_input, model = load_saved_results(path_to_params)
    gate_expander = expand_gate_from_saved_results(data_input, model, n_clusters_for_expansion)
    plot_and_save_expanded_data_for_samples(gate_expander, model, data_input, sample_idxs_to_plot)

    all_expanded_data = np.concatenate(gate_expander.expanded_data_per_sample)
    cell_level_labels = gate_expander.get_catted_cell_level_labels_of_expanded_data()

    catted_tr_data = np.concatenate(data_input.x_tr)
    plotter = DataAndGatesPlotterDepthOne(model, catted_tr_data)
    fig, axes = plotter.plot_data_with_gates(np.array(np.concatenate([data_input.y_tr[i] * torch.ones([data_input.x_tr[i].shape[0], 1]) for i in range(len(data_input.x_tr))])))
    
    size = 1000 * 1/catted_tr_data.shape[0]
    pos_cells = all_expanded_data[cell_level_labels == 1]
    neg_cells = all_expanded_data[cell_level_labels == 0]
    axes[0].scatter(pos_cells[:, 0], pos_cells[:, 1], color='r', s=size)
    axes[1].scatter(neg_cells[:, 0], neg_cells[:, 1], color='r', s=size)
    plt.savefig('expanded_data_with_all_data.png')
    
        
def main_kde_expansion(path_to_params):
    start_time = time.time()
    sample_idxs_to_plot = [0, 1, 2, 3, 4]
    step_size = .001
    sigma_thresh_factor = .5

    data_input, model, params = load_saved_results(path_to_params, ret_params_too=True)
    init_gate = model.get_gates()[0]
    gate_expander = KDEGateExpander(data_input.x_tr, init_gate, step_size=step_size, sigma_thresh_factor=sigma_thresh_factor)
    expanded_gates = gate_expander.expand_gates()
#    gate_expander = expand_gate_from_saved_results(data_input, model, n_clusters_for_expansion)
#    plot_and_save_expanded_data_for_samples(gate_expander, model, data_input, sample_idxs_to_plot)

    gate_expander.collect_expanded_cells_per_sample()

    # rerun logistic regressor to get the new accuracy after expansion
    expanded_gate_tree = [
        [
            ['D1', expanded_gates[0], expanded_gates[1]], 
            ['D2', expanded_gates[2], expanded_gates[3]]
        ]
    ]
    model.init_nodes(expanded_gate_tree)
    train_params = params['train_params']
    fit_classifier_params(model, data_input, train_params['learning_rate_classifier'], l1_reg_strength=train_params['l1_reg_strength'])
    expanded_output = model(data_input.x_tr, data_input.y_tr)
    y_true = data_input.y_tr
    y_pred = (expanded_output['y_pred'].cpu().detach().numpy() >= 0.5) * 1.0
    y_pred = y_pred.reshape(y_true.cpu().numpy().shape)
    acc_tr = sum(y_pred == y_true.cpu().numpy()) * 1.0 / y_true.shape[0]
    print('Tr acc/loss after expansion: %.4f, %.4f' %(acc_tr, expanded_output['log_loss']))

    
    expanded_output = model(data_input.x_te, data_input.y_te)
    y_true = data_input.y_te
    y_pred = (expanded_output['y_pred'].cpu().detach().numpy() >= 0.5) * 1.0
    y_pred = y_pred.reshape(y_true.cpu().numpy().shape)
    acc_te = sum(y_pred == y_true.cpu().numpy()) * 1.0 / y_true.shape[0]
    print('Te acc/loss after expansion: %.4f, %.4f' %(acc_te, expanded_output['log_loss']))
    

    all_expanded_data = np.concatenate(gate_expander.expanded_data_per_sample)
    print(all_expanded_data.shape, 'all_expanded_data')
    # get catted labels here

    cell_level_labels = get_catted_cell_level_labels_of_expanded_data(gate_expander.expanded_data_per_sample, data_input.y_tr)
    catted_tr_data = np.concatenate(data_input.x_tr)
    plotter = DataAndGatesPlotterDepthOne(model, catted_tr_data)
    fig, axes = plotter.plot_data_with_gates(np.array(np.concatenate([data_input.y_tr[i] * torch.ones([data_input.x_tr[i].shape[0], 1]) for i in range(len(data_input.x_tr))])))
    
    size = 1000 * 1/catted_tr_data.shape[0]
    pos_cells = all_expanded_data[cell_level_labels == 1, :]
    neg_cells = all_expanded_data[cell_level_labels == 0, :]
    axes[0].scatter(pos_cells[:, 0], pos_cells[:, 1], color='r', s=size)
    axes[1].scatter(neg_cells[:, 0], neg_cells[:, 1], color='r', s=size)
    plt.savefig('expanded_data_with_all_data.png')
    print('total time %.3f' %(time.time() - start_time))
    

def get_catted_cell_level_labels_of_expanded_data(expanded_data_per_sample, labels):
    cell_labels_per_sample = []
    for i, expanded_data in enumerate(expanded_data_per_sample):
        cell_labels = np.array(labels[i] * torch.ones([expanded_data.shape[0]]))
        #print(cell_labels.shape)
        cell_labels_per_sample.append(cell_labels)
    return np.concatenate(cell_labels_per_sample)


def expand_gate_from_saved_results(data_input, model, n_clusters_for_expansion):
    gate = model.get_gates()[0]
    gate_expander = KMeansGateExpander(n_clusters_for_expansion, data_input, gate)
    # creates a list of expanded data matching data_input.x_tr
    gate_expander.expand_data_past_gates(expand_tr_data_only=True)
    return gate_expander    


class KDEGateExpander:
    def __init__(self, data, init_gate, step_size=.05, 
        density_percent_thresh=.05, num_cells_for_kde=1e4,
        sigma_thresh_factor=1):
        self.data = data
        self.expanded_data_per_sample = []
        self.catted_data = self.init_catted_data(num_cells_for_kde)
        self.init_gate = init_gate
        self.step_size = step_size
        self.density_percent_thresh = density_percent_thresh
        self.sigma_thresh_factor = sigma_thresh_factor

    def init_catted_data(self, num_cells_for_kde):
        catted_data_all = np.concatenate(self.data)
        random_idxs = np.random.permutation(catted_data_all.shape[0])
        return catted_data_all[random_idxs]
        
    def expand_gates(self):
        self.fit_kde()
        self.find_max_density_in_init_gate()
        self.expanded_gate = copy.deepcopy(self.init_gate)
        # expansion for now just uses the midpoints, could also average over the sides of the box
        self.expand_lower_boundaries()
        self.expand_upper_boundaries()
        print('Before expansion: ', self.init_gate, 'After expansion: ', self.expanded_gate)
        return self.expanded_gate

    def fit_kde(self):
        # fit 2d kde to data (handle any subsampling oustide of this object)
        rule_of_thumb_bw_x = 1.06 * np.std(self.catted_data[:, 0]) * self.catted_data.shape[0] ** (-1/5)
        rule_of_thumb_bw_y = 1.06 * np.std(self.catted_data[:, 1]) * self.catted_data.shape[0] ** (-1/5)
        avg_rot_bw = (rule_of_thumb_bw_x + rule_of_thumb_bw_y)/2
        print('Bandwidth used is:%.3f' %avg_rot_bw)
        self.kde = KernelDensity(kernel='gaussian', bandwidth=avg_rot_bw).fit(self.catted_data)
        return self.kde

    # change to density threshold 
    def find_max_density_in_init_gate(self, grid_size=50):
        # TODO: fix so that its not other a straight diagonal line!!
        x_dims = np.linspace(self.init_gate[0], self.init_gate[1], num=grid_size).reshape(-1, 1)
        y_dims = np.linspace(self.init_gate[2], self.init_gate[3], num=grid_size).reshape(-1, 1)
        xx, yy = np.meshgrid(x_dims, y_dims)
        search_coords = np.stack((xx, yy), axis=-1).reshape(-1, 2)
        #search_coords = np.concatenate([x_dims, y_dims], axis=1)
        density = np.exp(self.kde.score_samples(search_coords))

        # test plot
#        sb.heatmap(density.reshape([]))
        

        std_density = np.std(density)
        mean_density = np.mean(density)
        max_density = np.max(density)
        print('Maximum of density inside un-expanded box: %.4f' %max_density)
        self.max_density_init_gate = max_density
        self.density_thresh = mean_density - std_density * self.sigma_thresh_factor
        if self.density_thresh < 0:
            self.density_thresh = 0
        
    def expand_lower_boundaries(self):
        # expand dim1 lower boundary
        not_converged = True
        while not_converged:
            midpoint_y = (self.expanded_gate[3] - self.expanded_gate[2])/2
            cur_density = np.exp(self.kde.score_samples(
                np.array([[self.expanded_gate[0], midpoint_y]])
            ))
            print('Dim1 lower, Cutoff: %.5f, Current Density: %.5f' %(self.density_thresh, cur_density))
            less_than_zero = self.expanded_gate[0] - self.step_size <= 0
            not_converged = (cur_density > self.density_thresh) and not less_than_zero
            #not_converged = (cur_density > self.max_density_init_gate * self.density_percent_thresh) and not less_than_zero
            if not_converged:
                self.expanded_gate[0] -= self.step_size
             

        # expand dim2 lower boundary
        not_converged = True
        while not_converged:
            midpoint_x = (self.expanded_gate[1] - self.expanded_gate[0])/2
            cur_density = np.exp(self.kde.score_samples(
                np.array([[midpoint_x, self.expanded_gate[2]]])
            ))
            print('Dim2 lower, Cutoff: %.5f, Current Density: %.5f' %(self.density_thresh, cur_density))
            less_than_zero = self.expanded_gate[2] - self.step_size <= 0
            not_converged = (cur_density > self.density_thresh) and not less_than_zero
            #not_converged = (cur_density > self.max_density_init_gate * self.density_percent_thresh) and not less_than_zero
            if not_converged:
                self.expanded_gate[2] -= self.step_size

    def expand_upper_boundaries(self):
        # expand dim1 upper boundary
        not_converged = True
        while not_converged:
            midpoint_y = (self.expanded_gate[3] - self.expanded_gate[2])/2
            cur_density = np.exp(self.kde.score_samples(
                np.array([[self.expanded_gate[1], midpoint_y]])
            ))
            print('Dim1 upper, Cutoff: %.5f, Current Density: %.5f' %(self.max_density_init_gate * self.density_percent_thresh, cur_density))
            greater_than_one = self.expanded_gate[1] + self.step_size >= 1
            not_converged = (cur_density > self.density_thresh) and not greater_than_one
            #not_converged = (cur_density > self.max_density_init_gate * self.density_percent_thresh) and not greater_than_one
            if not_converged:
                self.expanded_gate[1] += self.step_size
        
        # expand dim2 upper boundary 
        not_converged = True
        while not_converged:
            midpoint_x = (self.expanded_gate[1] - self.expanded_gate[0])/2
            cur_density = np.exp(self.kde.score_samples(
                np.array([[midpoint_x, self.expanded_gate[3]]])
            ))
            print('Dim2 upper, Cutoff: %.5f, Current Density: %.5f' %(self.max_density_init_gate * self.density_percent_thresh, cur_density))
            greater_than_one = self.expanded_gate[3] + self.step_size >= 1
            not_converged = (cur_density > self.density_thresh) and not greater_than_one
            #not_converged = (cur_density > self.max_density_init_gate * self.density_percent_thresh) and not greater_than_one
            if not_converged:
                self.expanded_gate[3] += self.step_size

    def get_expanded_data_new_samples(self, data):
        expanded_data = []
        for sample in data:
            idxs_init_gate = dh.filter_rectangle(sample,
                0, 1,
                self.init_gate[0], self.init_gate[1],
                self.init_gate[2], self.init_gate[3],
                return_idx=True
            )

            idxs_final_gate = dh.filter_rectangle(sample,
                0, 1,
                self.expanded_gate[0], self.expanded_gate[1],
                self.expanded_gate[2], self.expanded_gate[3],
                return_idx=True
            )

            expanded_cell_bool_idxs = ~idxs_init_gate & idxs_final_gate
            expanded_data.append(sample[expanded_cell_bool_idxs])
        return expanded_data

    def collect_expanded_cells_per_sample(self):
        
        for sample in self.data:
            idxs_init_gate = dh.filter_rectangle(sample,
                0, 1,
                self.init_gate[0], self.init_gate[1],
                self.init_gate[2], self.init_gate[3],
                return_idx=True
            )

            idxs_final_gate = dh.filter_rectangle(sample,
                0, 1,
                self.expanded_gate[0], self.expanded_gate[1],
                self.expanded_gate[2], self.expanded_gate[3],
                return_idx=True
            )

            expanded_cell_bool_idxs = ~idxs_init_gate & idxs_final_gate
            self.expanded_data_per_sample.append(sample[expanded_cell_bool_idxs])

class KMeansGateExpander:

    # for now only handles one gate, should handle multiple gates in general
    # for now only deals with tr data
    def __init__(self, k, data_input, gate, expand_tr_data_only=True):
        self.k = k
        self.data_input = data_input
        self.gate = gate
        self.expand_tr_data_only = expand_tr_data_only

    def expand_data_past_gates(self, expand_tr_data_only=True, random_state=0):
        self.expanded_data_per_sample = []
        self.clusterers_per_sample = []
        if not expand_tr_data_only:
            raise NotImplementedError('only tr data expansion implemented so far')

        for sample in self.data_input.x_tr:
            expanded_data, clusterer = self.expand_data_past_gates_single_sample(sample, random_state=0)
            self.expanded_data_per_sample.append(expanded_data)
            self.clusterers_per_sample.append(clusterer)

    def expand_data_past_gates_single_sample(self, sample, random_state=0):
        clusterer = KMeans(n_clusters=self.k, random_state=0).fit(sample)
        clusters = clusterer.cluster_centers_
        clusters_bool_idxs_in_gate = dh.filter_rectangle(clusters,
            0, 1,
            self.gate[0], self.gate[1],
            self.gate[2], self.gate[3],
            return_idx=True
        )
        clusters_idxs_in_gate = [idx for idx in range(clusters_bool_idxs_in_gate.shape[0]) if clusters_bool_idxs_in_gate[idx]]

        data_idxs_in_gate = dh.filter_rectangle(sample,
            0, 1,
            self.gate[0], self.gate[1],
            self.gate[2], self.gate[3],
            return_idx=True
        )

        cells_outside_gate_per_cluster = []
        for cluster_idx in clusters_idxs_in_gate:
            print(cluster_idx, clusters[cluster_idx])
            cells_in_cluster = clusterer.labels_ == cluster_idx
            cells_outside_gate = cells_in_cluster & ~data_idxs_in_gate.cpu().detach().numpy()
            cells_outside_gate_per_cluster.append(sample[cells_outside_gate])

        if len(cells_outside_gate_per_cluster) == 0:
            return np.zeros(sample.shape), clusterer
        return np.concatenate(cells_outside_gate_per_cluster), clusterer

    def get_catted_cell_level_labels_of_expanded_data(self):
        cell_labels_per_sample = []
        for i, expanded_data in enumerate(self.expanded_data_per_sample):
            cell_labels = np.array(self.data_input.y_tr[i] * torch.ones([expanded_data.shape[0]]))
            cell_labels_per_sample.append(cell_labels)
        return np.concatenate(cell_labels_per_sample)
    
        

            



def load_saved_results(path_to_params, ret_params_too=False):
    start_time = time.time()

    params = TransformParameterParser(path_to_params).parse_params()
    print(params)

    torch.manual_seed(params['random_seed'])
    np.random.seed(params['random_seed'])

    data_input = load_and_prepare_data_input(params)

    model = DepthOneModel([[['D1', 0, 0], ['D2', 0, 0]]], params['model_params'])
    model.load_state_dict(torch.load(os.path.join(params['save_dir'], 'model.pkl')))
    if ret_params_too:
        return data_input, model, params
    return data_input, model
    

def load_and_prepare_data_input(params):
    data_input = DataInput(params['data_params'])
    data_input.split_data()
    print('%d samples in the training data' %len(data_input.x_tr))

    with open(os.path.join(params['save_dir'], 'transformer.pkl'), 'rb') as f:
        data_transformer = pickle.load(f)

    # for debugging
    #params['transform_params']['cells_to_subsample'] = 2
    data_input.embed_data(\
        data_transformer,
        cells_to_subsample=params['transform_params']['cells_to_subsample'],
        use_labels_to_transform_data=params['transform_params']['use_labels_to_transform_data']
    ) 
    data_input.normalize_data()
    data_input.convert_all_data_to_tensors()
    return data_input

def plot_and_save_expanded_data_for_samples(gate_expander, model, data_input, sample_idxs_to_plot):
    for sample_idx in sample_idxs_to_plot:
        plotter = DataAndGatesPlotterDepthOne(model, np.concatenate(data_input.x_tr))
        plotter.plot_single_sample_with_gate(data_input.x_tr[sample_idx],
            data_input.idxs_tr[sample_idx], data_input.y_tr[sample_idx],
            plt.gca(),
            include_diagnostics=False
        )

        expanded_data = gate_expander.expanded_data_per_sample[sample_idx]
        if not(expanded_data.shape[0] == 0):
            print(expanded_data.shape)
            plotter.plot_single_sample_with_gate(expanded_data,
                data_input.idxs_tr[sample_idx], data_input.y_tr[sample_idx],
                plt.gca(), color='r', include_diagnostics=False
            )
        
        clusters = gate_expander.clusterers_per_sample[sample_idx].cluster_centers_
        plt.gca().scatter(clusters[:, 0], clusters[:, 1], color='b', s=2)
        plt.savefig('expanded_sample%d.png' %data_input.idxs_tr[sample_idx])
        plt.clf() 

if __name__ == '__main__':
    path_to_params = '../configs/umap_with_feat_diff_reg.yaml'
    main_kde_expansion(path_to_params)
