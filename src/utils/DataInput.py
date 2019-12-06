from sklearn.model_selection import train_test_split
from utils.utils_load_data import normalize_x_list
import pickle
import numpy as np
import torch
import os

class DataInput:
    
    def __init__(self, data_params):
        with open(data_params['path_to_x_list'], 'rb') as f:
            self.x_all = pickle.load(f)
        with open(data_params['path_to_labels'], 'rb') as f:
            self.y_all = pickle.load(f)
        assert(len(self.x_all) == len(self.y_all))
        self.data_params = data_params

    def split_data(self):
        self.x_tr, self.x_te, self.y_tr, self.y_te =\
            train_test_split(
                self.x_all, self.y_all, 
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

    def embed_data(self, transformer, cells_to_subsample=1e5, num_cells_for_transformer=1e10):
        self.x_tr_raw = self.x_tr
        self.x_te_raw = self.x_te
        if cells_to_subsample: 
            transformer.fit(self.out_of_place_shuffle(np.concatenate(self.x_tr))[0:int(num_cells_for_transformer)])
            self.x_tr = [transformer.transform(self.out_of_place_shuffle(tr_sample)[0:int(cells_to_subsample)]) for tr_sample in self.x_tr]
            self.x_te = [transformer.transform(self.out_of_place_shuffle(te_sample)[0:int(cells_to_subsample)]) for te_sample in self.x_te]
        else:
            transformer.fit(np.concatenate(self.x_tr)[0:])
            self.x_tr = [transformer.transform(tr_sample[0:]) for tr_sample in self.x_tr]
            self.x_te = [transformer.transform(te_sample[0:]) for te_sample in self.x_te]
        
        self.transformer = transformer
    
    def out_of_place_shuffle(self, np_arr):
        np.random.shuffle(np_arr)
        return np_arr
        
       
    def prepare_data_for_training(self):
        self.x_tr = self.convert_samples_to_tensors(self.x_tr)
        self.x_te = self.convert_samples_to_tensors(self.x_te)
        self.x_tr_raw = self.convert_samples_to_tensors(self.x_tr_raw)
        self.x_te_raw = self.convert_samples_to_tensors(self.x_te_raw)
        self.y_tr = self.convert_sample_to_tensor(self.y_tr)
        self.y_te = self.convert_sample_to_tensor(self.y_te)

    def convert_samples_to_tensors(self, samples):
        return [self.convert_sample_to_tensor(sample) for sample in samples]

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
