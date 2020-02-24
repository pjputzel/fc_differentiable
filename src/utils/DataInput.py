from sklearn.model_selection import train_test_split
from utils.utils_load_data import normalize_x_list
import utils.utils_load_data as dh
import pickle
import numpy as np
import torch
import os
import matplotlib.pyplot as plt

class DataInput:
    
    def __init__(self, data_params):
        with open(data_params['path_to_x_list'], 'rb') as f:
            self.x_all = pickle.load(f)
        with open(data_params['path_to_labels'], 'rb') as f:
            self.y_all = pickle.load(f)
        with open(data_params['path_to_idxs'], 'rb') as f:
            self.idxs_all = pickle.load(f)
    
        assert(len(self.x_all) == len(self.y_all))
        self.data_params = data_params

    def split_data(self, split_seed=None):
        if split_seed or split_seed == 0:
            self.idxs_all = np.arange(len(self.x_all))
            self.x_tr, self.x_te, self.y_tr, self.y_te, self.idxs_tr, self.idxs_te =\
                train_test_split(
                    self.x_all, self.y_all, self.idxs_all,
                    test_size=self.data_params['test_percent'],
                    random_state=split_seed
                )
            print(self.idxs_tr)
        else:
            self.x_tr, self.x_te, self.y_tr, self.y_te, self.idxs_tr, self.idxs_te =\
                train_test_split(
                    self.x_all, self.y_all, self.idxs_all,
                    test_size=self.data_params['test_percent']
                )

    def normalize_data(self):
        try:
            _ = self.x_tr
        except:
            raise RuntimeError('Split must be called before normalize!')
        tr_data_normalized, self.offset, self.scale = normalize_x_list(self.x_tr)
        te_data_normalized, _, _ = normalize_x_list(self.x_te, offset=self.offset, scale=self.scale)
        self.x_tr = tr_data_normalized
        self.x_te = te_data_normalized
        
        #self.x_tr_raw, self.offset_raw, self.scale_raw = normalize_x_list(self.x_tr_raw)
        #self.x_te_raw, _, _ = normalize_x_list(self.x_te_raw, self.offset_raw, self.scale_raw)
    
    def embed_data(self, transformer, cells_to_subsample=1e5, use_labels_to_transform_data=False):
        self.x_tr_raw = self.x_tr
        self.x_te_raw = self.x_te
        if cells_to_subsample: 
            shuffle_idxs_per_sample = [np.random.permutation(sample.shape[0]) for sample in self.x_tr]
            self.untransformed_matched_x_tr = [tr_sample[shuffle_idxs_per_sample[sample_idx]][0:int(cells_to_subsample)] for sample_idx, tr_sample in enumerate(self.x_tr)]
            self.x_tr = [transformer.transform(tr_sample[shuffle_idxs_per_sample[sample_idx]][0:int(cells_to_subsample)]) for sample_idx, tr_sample in enumerate(self.x_tr)]
            #self.x_tr = [transformer.transform(self.out_of_place_shuffle(tr_sample)[0:int(cells_to_subsample)]) for tr_sample in self.x_tr]
            self.x_te = [transformer.transform(self.out_of_place_shuffle(te_sample)[0:int(cells_to_subsample)]) for te_sample in self.x_te]
        else:
            self.x_tr = [transformer.transform(tr_sample[0:]) for tr_sample in self.x_tr]
            self.x_te = [transformer.transform(te_sample[0:]) for te_sample in self.x_te]
        
        self.transformer = transformer

    def embed_data_and_fit_transformer(self, transformer, cells_to_subsample=1e5, num_cells_for_transformer=1e10, use_labels_to_transform_data=False):
        self.x_tr_raw = self.x_tr
        self.x_te_raw = self.x_te
        cell_level_labels = np.concatenate([np.ones(x.shape[0]) * label for x, label in zip(self.x_tr, self.y_tr)])
        if cells_to_subsample:
            permute_idxs = np.random.permutation(cell_level_labels.shape[0])
            labels_keyword_argument = cell_level_labels[permute_idxs][0:int(num_cells_for_transformer)] if use_labels_to_transform_data else None
            transformer.fit(np.concatenate(self.x_tr)[permute_idxs][0:int(num_cells_for_transformer)], y=labels_keyword_argument)

            shuffle_idxs_per_sample = [np.random.permutation(sample.shape[0]) for sample in self.x_tr]
            self.untransformed_matched_x_tr = [tr_sample[shuffle_idxs_per_sample[sample_idx]][0:int(cells_to_subsample)] for sample_idx, tr_sample in enumerate(self.x_tr)]
            self.x_tr = [transformer.transform(tr_sample[shuffle_idxs_per_sample[sample_idx]][0:int(cells_to_subsample)]) for sample_idx, tr_sample in enumerate(self.x_tr)]
            self.x_te = [transformer.transform(self.out_of_place_shuffle(te_sample)[0:int(cells_to_subsample)]) for te_sample in self.x_te]

        else:
            labels_keyword_argument = cell_level_labels if use_labels_to_transform_data else None
            transformer.fit(np.concatenate(self.x_tr)[0:], y=labels_keyword_argument)
            self.x_tr = [transformer.transform(tr_sample[0:]) for tr_sample in self.x_tr]
            self.x_te = [transformer.transform(te_sample[0:]) for te_sample in self.x_te]
        
        self.transformer = transformer
    
    def out_of_place_shuffle(self, np_arr):
        np.random.shuffle(np_arr)
        return np_arr
        
       
    def convert_all_data_to_tensors(self):
        self.x_tr = self.convert_samples_to_tensors(self.x_tr)
        self.x_te = self.convert_samples_to_tensors(self.x_te)
        self.x_tr_raw = self.convert_samples_to_tensors(self.x_tr_raw)
        self.x_te_raw = self.convert_samples_to_tensors(self.x_te_raw)
        self.y_tr = self.convert_sample_to_tensor(self.y_tr)
        self.y_te = self.convert_sample_to_tensor(self.y_te)

    def convert_samples_to_tensors(self, samples):
        return [self.convert_sample_to_tensor(sample) for sample in samples]

    def filter_data_inside_first_model_gate(self, model):
        gate = model.get_gates()[0]
        gate = [g if g > 0 else 0 for g in gate]
        idxs_in_gate_per_sample = [dh.filter_rectangle(x, 0, 1, gate[0], gate[1], gate[2], gate[3], return_idx=True) for x in self.x_tr]
        #idxs_in_gate = dh.filter_rectangle(
        #    self.x_tr, 0, 1, gate[0], gate[1], gate[2], gate[3],
        #    return_idx=True
        #)
        self.unfiltered_by_model_gates_x_tr = self.x_tr
        self.x_tr = [x_tr[idxs_in_gate].detach().numpy() for x_tr, idxs_in_gate in zip(self.x_tr, idxs_in_gate_per_sample)]
        #put in one dummy data point so umap can work
        self.x_tr = [x_tr if x_tr.shape[0] > 0 else np.array([[0, 0]]) for x_tr in self.x_tr]
        self.y_tr = self.y_tr.detach().numpy()
#        self.x_tr = self.untransformed_matched_x_tr[idxs]
        self.filtered_idxs_per_sample = idxs_in_gate_per_sample

    # TODO add cuda functionality here
    def convert_sample_to_tensor(self, sample):
        return torch.tensor(sample, dtype=torch.float32)
   
    def save_transformer(self, savedir):
        savepath = os.path.join(savedir, 'transformer.pkl')
        with open(savepath, 'wb') as f:
            pickle.dump(self.transformer, f)
    
    def get_pos_cat_tr_data(self):
        return np.concatenate([data for i, data in enumerate(self.x_tr) if self.y_tr[i] == 1])

    def get_neg_cat_tr_data(self):
        return np.concatenate([data for i, data in enumerate(self.x_tr) if self.y_tr[i] == 0])
