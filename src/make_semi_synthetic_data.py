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
import utils.utils_load_data as dh
from train_UMAP import run_train_model
import torch
import os
import time
from copy import deepcopy


# using data in the folder data_for_UMAP_with_filtering
# if a different data version is used this dictionary will have to change
DIM_IDXS_TO_GATE = {
        (0, 1): 'gate0',
        (3, 4): 'gate1',
        (0, 2): 'gate2',
        (6, 7): 'gate3',
        (11, 8): 'gate4',
        (5, 0): 'CD22_fake_gate'

        # ... fill me in correctly maps true idxs in the umap data to the matching gate name
        # note that once this is filled in we should also be able to write the filter functions
        # correctly over the correct dims for each gate
    }

GATE_NAME_TO_DIMS = {
    'gate0': (0, 1),
    'gate1': (3, 4),
    'gate2': (0, 2),
    'gate3': (6, 7),
    'gate4': (11, 8),
    'CD22_fake_gate': (5, 0)
}

class SemiSynthDataGeneratorFromTwoRepresentativeSamples:
    
    def __init__(self, rep_pos_sample, rep_neg_sample, shift_means, shift_vars, dims_to_shift, n_samples_total):
        self.rep_pos_sample = rep_pos_sample
        self.rep_neg_sample = rep_neg_sample
        self.shift_means = shift_means
        self.shift_vars = shift_vars 
        self.dims_to_shift = dims_to_shift
        self.num_shifts = dims_to_shift.shape[0]
        self.n_samples_total = n_samples_total
        self.shifted_pos_samples = []
        self.shifted_neg_samples = []
        self.all_samples = []
        self.init_idxs_within_gates()
    
        if not ((self.dims_to_shift.shape[0] == shift_means.shape[0]) and (self.dims_to_shift.shape[0] == shift_vars.shape[0])):
            raise ValueError('The number of means/variance matrices must match the number of dimensions to apply shifts')
        if not self.n_samples_total % 2 == 0:
            raise ValueError('The total number of samples must be divisible by two.')
    
    def init_idxs_within_gates(self):
        #applying gate functions to pos/neg sample per shift to get the matching idxs within a gate for each different dim_to_shift
        #result should be that there are self.idxs_within_gates_pos (*_neg) with entry per shift dim containing data within the corresponding gate from pb1 html file.
        self.idxs_within_gates_pos = []
        self.idxs_within_gates_neg = []
        for dim_to_shift in self.dims_to_shift:
            # creating a mapping from dims to shift to the correct filtering to do
            # map tuple of the idxs to correct filtering in a dict
            self.idxs_within_gates_pos.append(self.filter_to_dim_to_shift(dim_to_shift, pos=True))
            self.idxs_within_gates_neg.append(self.filter_to_dim_to_shift(dim_to_shift, pos=False))
                
    def filter_to_dim_to_shift(self, dim_to_shift, pos=True):
        if pos:
            data = self.rep_pos_sample
        else:
            data = self.rep_neg_sample

        gate_name = DIM_IDXS_TO_GATE[(dim_to_shift[0], dim_to_shift[1])]
        if gate_name == 'gate1':
            idxs = self.get_idxs_in_gate1(data, dim_to_shift)
        elif gate_name == 'gate2':
            idxs = self.get_idxs_in_gate2(data, dim_to_shift)
        elif gate_name == 'gate3':
            idxs = self.get_idxs_in_gate3(data, dim_to_shift)
        elif gate_name == 'gate4': 
            idxs = self.get_idxs_in_gate4(data, dim_to_shift)
        elif gate_name == 'CD22_fake_gate':
            idxs = data > -1 # just getting all idxs since no filtering for this one
        
        return idxs
    
    def get_idxs_in_gate0(self, data, dim_to_shift):
        idxs = dh.filter_slope(data, dim_to_shift[0], dim_to_shift[1],
                2048, 4096, 2048, 2560,
                return_idx=True
            )
        return idxs

    def get_idxs_in_gate1(self, data, dim_to_shift):
        idxs_gate_0 = self.get_idxs_in_gate0(data, GATE_NAME_TO_DIMS['gate0'])
        idxs_gate_1 = dh.filter_rectangle(
                        data, dim_to_shift[0], dim_to_shift[1],
                        102, 921, 2048, 3891,
                        return_idx=True
        ) 
        print(idxs_gate_0.shape, idxs_gate_1.shape)
#        idxs_in_both = [i for i, idx in enumerate(idxs_gate_1) if idx in idxs_gate_0[i]]
        idxs_in_both = idxs_gate_0  & idxs_gate_1
        return idxs_in_both

    def get_idxs_in_gate2(self, data, dim_to_shift):
        idxs_gate_1 = self.get_idxs_in_gate1(data, GATE_NAME_TO_DIMS['gate1'])
        idxs_gate_2 = dh.filter_rectangle(
                        data, dim_to_shift[0], dim_to_shift[1],
                        921, 2150, 102, 921,
                        return_idx=True
        ) 
        idxs_in_both = idxs_gate_1 & idxs_gate_2
        return idxs_in_both
        
    def get_idxs_in_gate3(self, data, dim_to_shift):
        idxs_gate_2 = self.get_idxs_in_gate2(data, GATE_NAME_TO_DIMS['gate2'])
        idxs_gate_3 = dh.filter_rectangle(
                        data, dim_to_shift[0], dim_to_shift[1],
                        1638, 3891, 2150, 3891,
                        return_idx=True
        ) 
        idxs_in_both = idxs_gate_2 & idxs_gate_3
        return idxs_in_both

    def get_idxs_in_gate4(self, data, dim_to_shift):
        idxs_gate_3 = self.get_idxs_in_gate3(data, GATE_NAME_TO_DIMS['gate3'])
        idxs_gate_4 = dh.filter_rectangle(
                        data, dim_to_shift[0], dim_to_shift[1],
                        0., 1228,0, 1843,
                        return_idx=True
        ) 
        idxs_in_both = idxs_gate_3 & idxs_gate_4
        return idxs_in_both

    def produce_all_shifted_data(self):
        for new_sample_idx in range(self.n_samples_total//2):
            pos_shifted_sample, neg_shifted_sample = self.apply_all_random_shifts_once()
            self.shifted_pos_samples.append(pos_shifted_sample)
            self.shifted_neg_samples.append(neg_shifted_sample)

    def apply_all_random_shifts_once(self):
        cur_data_pos = None
        cur_data_neg = None
        for shift_index in range(self.dims_to_shift.shape[0]):
            cur_data_pos = self.apply_single_dim_shift(cur_data_pos, shift_index, data_is_pos=True)
            cur_data_neg = self.apply_single_dim_shift(cur_data_neg, shift_index, data_is_pos=False)

        return cur_data_pos, cur_data_neg

    def apply_single_dim_shift(self, data, shift_index, data_is_pos=True):
        if data_is_pos:
            idxs_within_gates = self.idxs_within_gates_pos[shift_index]
        else:
            idxs_within_gates = self.idxs_within_gates_neg[shift_index]

        if not type(data) == np.ndarray:
            if not data:
                if data_is_pos:
                    data = deepcopy(self.rep_pos_sample)
                else:
                    data = deepcopy(self.rep_neg_sample)

        shift = np.random.multivariate_normal(
                    self.shift_means[shift_index],
                    self.shift_vars[shift_index]
                )
        print(shift)
        print(np.sum(idxs_within_gates), 'meow')
        idxs_within_gates = np.array([i for i in range(idxs_within_gates.shape[0]) if idxs_within_gates[i]])
        data_within_gate = data[idxs_within_gates][:, self.dims_to_shift[shift_index]]
        data[idxs_within_gates.reshape(-1, 1), self.dims_to_shift[shift_index]] =  data_within_gate + shift
        return data
            

        
class SemiSynthDataGenerator:
    
    def __init__(self, shift_means, shift_vars, dims_to_shift):
        self.shift_means = shift_means
        self.shift_vars = shift_vars
        self.dims_to_shift = dims_to_shift
 
        if not ((self.dims_to_shift.shape[0] == shift_means.shape[0]) and (self.dims_to_shift.shape[0] == shift_vars.shape[0])):
            raise ValueError('The number of means/variance matrices must match the number of dimensions to apply shifts')


    def get_idxs_in_gate0(self, data, dim_to_shift):
        idxs = dh.filter_slope(data, dim_to_shift[0], dim_to_shift[1],
                2048, 4096, 2048, 2560,
                return_idx=True
            )
        return idxs

    def get_idxs_in_gate1(self, data, dim_to_shift):
        idxs_gate_0 = self.get_idxs_in_gate0(data, GATE_NAME_TO_DIMS['gate0'])
        idxs_gate_1 = dh.filter_rectangle(
                        data, dim_to_shift[0], dim_to_shift[1],
                        102, 921, 2048, 3891,
                        return_idx=True
        ) 
        idxs_in_both = idxs_gate_0  & idxs_gate_1
        return idxs_in_both

    def get_idxs_in_gate2(self, data, dim_to_shift):
        idxs_gate_1 = self.get_idxs_in_gate1(data, GATE_NAME_TO_DIMS['gate1'])
        idxs_gate_2 = dh.filter_rectangle(
                        data, dim_to_shift[0], dim_to_shift[1],
                        921, 2150, 102, 921,
                        return_idx=True
        ) 
        idxs_in_both = idxs_gate_1 & idxs_gate_2
        return idxs_in_both
        
    def get_idxs_in_gate3(self, data, dim_to_shift):
        idxs_gate_2 = self.get_idxs_in_gate2(data, GATE_NAME_TO_DIMS['gate2'])
        idxs_gate_3 = dh.filter_rectangle(
                        data, dim_to_shift[0], dim_to_shift[1],
                        1638, 3891, 2150, 3891,
                        return_idx=True
        ) 
        idxs_in_both = idxs_gate_2 & idxs_gate_3
        return idxs_in_both

    def get_idxs_in_gate4(self, data, dim_to_shift):
        idxs_gate_3 = self.get_idxs_in_gate3(data, GATE_NAME_TO_DIMS['gate3'])
        idxs_gate_4 = dh.filter_rectangle(
                        data, dim_to_shift[0], dim_to_shift[1],
                        0., 1228,0, 1843,
                        return_idx=True
        ) 
        idxs_in_both = idxs_gate_3 & idxs_gate_4
        return idxs_in_both

    def apply_shifts_to_samples(self, samples):
        shifted_samples = []
        for sample in samples:
            shifted_samples.append(self.apply_shifts_to_single_sample(sample))
        return shifted_samples

    def apply_shifts_to_single_sample(self, sample):
        cur_data = sample
        idxs_within_gate = self.get_idxs_within_gate_single_sample(sample)
        for shift_index in range(self.dims_to_shift.shape[0]):
            sample = self.apply_single_dim_shift(sample, shift_index, idxs_within_gate[shift_index])
        return sample

    def get_idxs_within_gate_single_sample(self, sample):
        idxs_within_gates = []
        for dim_to_shift in self.dims_to_shift:
            idxs_within_gates.append(self.filter_to_dim_to_shift(dim_to_shift, sample))
        return idxs_within_gates

    def filter_to_dim_to_shift(self, dim_to_shift, data):

        gate_name = DIM_IDXS_TO_GATE[(dim_to_shift[0], dim_to_shift[1])]
        if gate_name == 'gate1':
            idxs = self.get_idxs_in_gate1(data, dim_to_shift)
        elif gate_name == 'gate2':
            idxs = self.get_idxs_in_gate2(data, dim_to_shift)
        elif gate_name == 'gate3':
            idxs = self.get_idxs_in_gate3(data, dim_to_shift)
        elif gate_name == 'gate4': 
            idxs = self.get_idxs_in_gate4(data, dim_to_shift)
        elif gate_name == 'CD22_fake_gate':
            idxs = data[:, 0] > -1e10  # to collect all boolean idxs since no filtering for CD22 version of dataset
        
        return idxs

    def apply_single_dim_shift(self, sample, shift_index, idxs_within_gate):

        shift = np.random.multivariate_normal(
                    self.shift_means[shift_index],
                    self.shift_vars[shift_index]
                )
        print(shift)
        print(np.sum(idxs_within_gate), 'meow')
        if np.sum(idxs_within_gate) == 0:
            print('no cells within the gates!')
            return sample
        idxs_within_gate = np.array([i for i in range(idxs_within_gate.shape[0]) if idxs_within_gate[i]])
        data_within_gate = sample[idxs_within_gate][:, self.dims_to_shift[shift_index]]
        sample[idxs_within_gate.reshape(-1, 1), self.dims_to_shift[shift_index]] =  data_within_gate + shift
        return sample



def load_data_input(path_to_params):
    params = TransformParameterParser(path_to_params).parse_params()
    print(params)

    set_random_seeds(params)

    data_input = DataInput(params['data_params'])
    data_input.split_data()
    return data_input

def get_representative_pos_and_negative_samples(path_to_params, rep_pos_name, rep_neg_name):
    data_input = load_data_input(path_to_params)
    rep_pos_index = data_input.idxs_all.index(rep_pos_name)
    rep_neg_index = data_input.idxs_all.index(rep_neg_name)
    pos_sample, neg_sample = data_input.x_all[rep_pos_index], data_input.x_all[rep_neg_index]
    return pos_sample, neg_sample

def set_random_seeds(params):
    torch.manual_seed(params['random_seed'])
    np.random.seed(params['random_seed'])

def main_dev_data_shift():
    path_to_params = '../configs/umap_with_feat_diff_reg.yaml'    
    shift1 = 2000
    shift2 = 0
    var = 200
    shift_along = 'CD22'
    
    # for CD22 shifting (make sure shift2=0)
    shift_means = np.array([[shift1, 0], [shift2, 0]])
    shift_vars = np.array([var*np.eye(2), var*np.eye(2)])
    dims_to_shift = np.array([[5, 0], [6, 7]])
    data_input = load_data_input(path_to_params)

#    # for CD5/CD79b shifting
#    shift_means = np.array([[0, shift1], [shift2, 0]])
#    shift_vars = np.array([var*np.eye(2), var*np.eye(2)])
#    dims_to_shift = np.array([[11, 8], [6, 7]])
#    data_input = load_data_input(path_to_params)
   
    shifted_data_generator = SemiSynthDataGenerator(shift_means, shift_vars, dims_to_shift)
    shifted_data = shifted_data_generator.apply_shifts_to_samples(data_input.x_all)

    all_data = shifted_data
    all_labels = data_input.y_all
    all_names = data_input.sample_names_all
    
    with open('../data/semi_synth/all_data_%d_%d_%s.pkl' %(shift1, shift2, shift_along), 'wb') as f:
        pickle.dump(all_data, f)


    with open('../data/semi_synth/all_labels_%d_%d_%s.pkl' %(shift1, shift2, shift_along), 'wb') as f:
        pickle.dump(all_labels, f)

    with open('../data/semi_synth/all_names_%d_%d_%s.pkl' %(shift1, shift2, shift_along), 'wb') as f:
        pickle.dump(all_names, f)
    pos_idxs = [i for i in range(len(data_input.y_all)) if data_input.y_all[i]]
    idx_to_plot = 3
    shifted_pos_sample = shifted_data[pos_idxs[idx_to_plot]]
    print('plotting %s' %(data_input.sample_names_all[pos_idxs[idx_to_plot]]))

    plt.scatter(shifted_pos_sample[:, dims_to_shift[0][0]], shifted_pos_sample[:, dims_to_shift[0][1]], s=.01)
    plt.savefig('testing_semi_synth.png')
    plt.clf()
    plt.scatter(shifted_pos_sample[:, dims_to_shift[1][0]], shifted_pos_sample[:, dims_to_shift[1][1]], s=.01)
    plt.savefig('testing_semi_synth2.png')

def main_two_representative_samples():
    path_to_params = '../configs/umap_with_feat_diff_all.yaml'
    rep_pos_name = 9343
    rep_neg_name = 7495
    shift1 = 0
    shift2 = -0
    var = .5
    shift_means = np.array([[shift1, 0], [0, shift2]])
    shift_vars = np.array([var*np.eye(2), var*np.eye(2)])
    dims_to_shift = np.array([[11, 8], [6, 7]])
    n_samples_total = 40 #change to 40 after debugging

    rep_pos_sample, rep_neg_sample = get_representative_pos_and_negative_samples(path_to_params, rep_pos_name, rep_neg_name)
    data_generator = SemiSynthDataGenerator(rep_pos_sample, rep_neg_sample, shift_means, shift_vars, dims_to_shift, n_samples_total)
    data_generator.produce_all_shifted_data()
    

    all_data = data_generator.shifted_pos_samples + data_generator.shifted_neg_samples
    all_labels = np.concatenate([np.ones(len(data_generator.shifted_pos_samples)), np.zeros(len(data_generator.shifted_neg_samples))])
    print(all_labels)
    all_names = np.arange(len(all_data))
    
    with open('../data/semi_synth/all_data_%d_%d.pkl' %(shift1, shift2), 'wb') as f:
        pickle.dump(all_data, f)


    with open('../data/semi_synth/all_labels_%d_%d.pkl' %(shift1, shift2), 'wb') as f:
        pickle.dump(all_labels, f)

    with open('../data/semi_synth/all_names_%d_%d.pkl' %(shift1, shift2), 'wb') as f:
        pickle.dump(all_names, f)

    print(len(data_generator.shifted_neg_samples), len(data_generator.shifted_pos_samples))
    plt.scatter(data_generator.shifted_pos_samples[0][:, 11], data_generator.shifted_pos_samples[0][:, 8], s=.01)
    plt.savefig('testing_semi_synth.png')
    plt.clf()
    plt.scatter(data_generator.shifted_pos_samples[0][:, 6], data_generator.shifted_pos_samples[0][:, 7], s=.01)
    plt.savefig('testing_semi_synth2.png')

if __name__ == '__main__':
    main_dev_data_shift()
