import prim
import pandas as pd
import numpy as np
import numpy.matlib
from collections import namedtuple as NamedTuple
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

class GateInitializerPrimKDE:
    def __init__(self, x_tr, gate_init_params):
        self.gate_init_params = gate_init_params
        self.x_tr = x_tr

    
    @staticmethod
    # for loading a saved model state_dict use these
    # fake gates to init the model and then load the
    # state dict to get the correct gate params
    # also can be used for debugging other parts
    # of the data pipeline
    def get_fake_init_gates(num_gates):
        fake_gates = []
        for g in range(num_gates):
            fake_gates.append(
                [   
                    ['D1', 0., 0.],
                    ['D2', 0., 0.]   
                ]
            )
        return fake_gates
                

    def initialize_gates(self):
        self.compute_box_memberships()
        self.init_gates_per_box(self.box_memberships_tr)
        self.construct_init_gate_tree()
        return self.init_gates
    


    def compute_box_memberships(self, kde_noise_variance=.0001):
        self.catted_x_tr = np.concatenate(self.x_tr)
        kde_data = self.catted_x_tr + np.random.multivariate_normal(np.zeros(self.catted_x_tr.shape[1]), kde_noise_variance * np.eye(2)) #prevent singular matrix apparently some datapoints are identical/overplotted
        kde_data = kde_data.reshape([kde_data.shape[1], kde_data.shape[0]])
        kde = gaussian_kde(kde_data)

        grid_for_prim = self.get_grid_for_prim()
        density_estimate = kde(grid_for_prim.reshape([grid_for_prim.shape[1], grid_for_prim.shape[0]]))
        max_density = np.amax(density_estimate)
        print(density_estimate[0:20])
        primmer = prim.Prim(
            pd.DataFrame(grid_for_prim), 
            density_estimate,
            threshold=max_density * self.gate_init_params['prim_threshold_percent']
        )
        #for i in range(self.gate_init_params['n_boxes']):
            #primmer.find_box()
        primmer.find_all()
        print(primmer.limits, 'bark')
        box_memberships_tr = self.get_box_memberships_tr(primmer._boxes)
        self.box_memberships_tr = box_memberships_tr

    def get_grid_for_prim(self):
        x, y = np.linspace(0, 1, self.gate_init_params['prim_grid_size']), np.linspace(0, 1, self.gate_init_params['prim_grid_size'])
        x, y = np.meshgrid(x, y)
        return np.concatenate([x.reshape([-1, 1]), y.reshape([-1, 1])], axis=1)  
 
    def get_box_memberships_tr(self, boxes):
        box_memberships = []
        for box in boxes:
            box_idxs = self.get_box_idxs_tr(box)
        box_memberships.append(box_idxs)
        return box_memberships
 
    def get_box_idxs_tr(self, box):
        print(box, 'meow')
        return 'meow'

    def init_gates_per_box(self, box_memberships_tr): 
        init_gates = []
        boxes = np.unique(box_memberships_tr)
        for box in boxes:
            # can add percentile low/high here if needed to gate init params
            init_gates.append(self.get_single_gate(box))
        self.init_gates = init_gates
        
        
    def get_single_gate(self, box, percentile_low=.20, percentile_high=.80):
        box_data = self.catted_x_tr[self.box_memberships_tr == box]
        sorted_x = np.sort(box_data[:, 0])
        sorted_y = np.sort(box_data[:, 1])
        low_idx = int(percentile_low * box_data.shape[0])
        high_idx = int(percentile_high * box_data.shape[0])
        gate = [sorted_x[low_idx], sorted_x[high_idx], 
                sorted_y[low_idx], sorted_y[high_idx]]
        return gate

    def construct_init_gate_tree(self):
        self.init_gate_tree = []
        for gate in self.init_gates:
            self.init_gate_tree.append(
                [[u'D1', gate[0], gate[1]], [u'D2', gate[2], gate[3]]]
            )
        return self.init_gate_tree




    def plot_init_gate_tree_with_data(self):
        
        cm = plt.get_cmap('hsv')
        self.colors = cm(np.linspace(0, 1, len(self.init_gate_tree)))
        self.plot_boxed_data(plt.gca())
        for g, gate in enumerate(self.init_gate_tree):
            self.plot_gate(plt.gca(), gate, color=self.colors[g])
    
    def plot_boxed_data(self, axis):
        for box in range(self.gate_init_params['n_boxes']):
            box_data = self.catted_x_tr[self.box_memberships_tr == box]
            plt.scatter(box_data[:, 0], box_data[:, 1], c=np.matlib.repmat(self.colors[-1 + box], box_data.shape[0], 1))
            

    def get_gate_named_tuple(self, gate):
        gate_n_tuple = NamedTuple('gate', ['low1', 'low2', 'upp1', 'upp2'])
        gate_n_tuple.low1, gate_n_tuple.low2 = gate[0][1], gate[1][1]
        gate_n_tuple.upp1, gate_n_tuple.upp2 = gate[0][2], gate[1][2]
        return gate_n_tuple

    def plot_gate(self, axis, gate, color='g', lw=3, dashes=(None, None), label=None):
        gate = self.get_gate_named_tuple(gate)
        axis.plot([gate.low1, gate.low1], [gate.low2, gate.upp2], c=color, 
            label=label, dashes=dashes, linewidth=lw)
        axis.plot([gate.low1, gate.upp1], [gate.low2, gate.low2], c=color, 
            dashes=dashes, linewidth=lw)
        axis.plot([gate.upp1, gate.upp1], [gate.low2, gate.upp2], c=color, 
            dashes=dashes, linewidth=lw)
        axis.plot([gate.upp1, gate.low1], [gate.upp2,gate.upp2], c=color, 
            dashes=dashes, linewidth=lw)
        return axis



