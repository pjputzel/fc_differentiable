from sklearn.cluster import KMeans
import numpy as np
import numpy.matlib
from collections import namedtuple as NamedTuple
import matplotlib.pyplot as plt

#TODO: add code for heuristic as well

class GateInitializer:
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
        self.compute_cluster_memberships()
        self.init_gates_per_cluster(self.cluster_memberships_tr)
        self.construct_init_gate_tree()
        return self.init_gates
    
    def initialize_gates_heuristic(self):
        self.init_gates = RepeatedHeuristicInitializer(self.gate_init_params['heuristic_params']).init_gates()
        return self.init_gates


    def compute_cluster_memberships(self):
        self.catted_x_tr = np.concatenate(self.x_tr)
        kmeans = KMeans(n_clusters=self.gate_init_params['n_clusters'])
        cluster_memberships_tr = kmeans.fit_predict(self.catted_x_tr)
#        cluster_memberships_tr = kmeans.predict(self.catted_x_tr) 
        self.cluster_memberships_tr = cluster_memberships_tr
    
    def init_gates_per_cluster(self, cluster_memberships_tr): 
        init_gates = []
        clusters = np.unique(cluster_memberships_tr)
        for cluster in clusters:
            # can add percentile low/high here if needed to gate init params
            init_gates.append(self.get_single_gate(cluster))
        self.init_gates = init_gates
        
        
    def get_single_gate(self, cluster, percentile_low=.20, percentile_high=.80):
        cluster_data = self.catted_x_tr[self.cluster_memberships_tr == cluster]
        sorted_x = np.sort(cluster_data[:, 0])
        sorted_y = np.sort(cluster_data[:, 1])
        low_idx = int(percentile_low * cluster_data.shape[0])
        high_idx = int(percentile_high * cluster_data.shape[0])
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
        self.plot_clustered_data(plt.gca())
        for g, gate in enumerate(self.init_gate_tree):
            self.plot_gate(plt.gca(), gate, color=self.colors[g])
    
    def plot_clustered_data(self, axis):
        for cluster in range(self.gate_init_params['n_clusters']):
            cluster_data = self.catted_x_tr[self.cluster_memberships_tr == cluster]
            plt.scatter(cluster_data[:, 0], cluster_data[:, 1], c=np.matlib.repmat(self.colors[-1 + cluster], cluster_data.shape[0], 1))
            

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








