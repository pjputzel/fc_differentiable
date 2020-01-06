import torch
import numpy as np
from collections import namedtuple
import utils.utils_load_data as dh
import torch.nn.functional as F
from utils.DepthOneModel import DepthOneModel
from utils.bayes_gate import ModelTree
import seaborn as sb
import matplotlib.pyplot as plt
import matplotlib

class DataAndGatesPlotter():

    '''
    Class to handle plotting the data and gates

    attributes:
        model: the model whose gates to plot
        data: the data to plot
        filtered_data: data filtered along the tree structure 
                        defined by model in depth first order
        dims: the indexes into self.data for which two dimensions
                a given node uses
    '''
    def __init__(self, model, data, is_4chain=False, color=None):
        self.model = model
        self.data = data
        self.gates = self.construct_gates()
        self.filtered_data = self.filter_data_by_model_gates()
        self.dims = self.get_dims()
        self.color = color

        # modification needed to plot reference tree objects
        #ids2features = self.model.referenceTree.ids2features
        #self.feature_names = [(ids2features[dim1], ids2features[dim2]) 
        #        for (dim1, dim2) in self.dims]



    '''
    loads the nodes gate params into a namedtuple
    '''
    @staticmethod
    def get_gate(node):
        gate = ModelTree.get_gate(node)
        return gate

    @staticmethod
    def get_dim_single_node(node):
        return (node.gate_dim1, node.gate_dim2)

    def get_dims(self):
        dims, _ = self.apply_function_depth_first(
                DataAndGatesPlotter.get_dim_single_node
               )
        return dims
            


    '''
    filters data using the gate from the input node
    '''
    def filter_data_at_single_node(self, data, node, return_idxs=False):
        gate = DataAndGatesPlotter.get_gate(node)
        return_value = dh.filter_rectangle(
                data, node.gate_dim1, 
                node.gate_dim2, gate.low1, gate.upp1, 
                gate.low2, gate.upp2,
                return_idx=return_idxs
        )
        
        return return_value

    def construct_gates(self):
        return self.apply_function_depth_first(
                    DataAndGatesPlotter.get_gate
                )[0]

    def filter_data_by_model_gates(self):
        # Pass a dummy function to just compute filtered_data
        _, filtered_data = self.apply_function_depth_first(lambda x, y: None,
                function_uses_data=True)
        return filtered_data

    '''
    applies the given function to each node in 
    the model tree in a depth first order, currently only
    implemented for a chain/line graph

    param function: the function to apply at each node
    
    returns output: A list of results from applying the function
                    to each node in depth first order.

    returns filtered_data: The filtered data at each node in depth first
                           order
    '''
    def apply_function_depth_first(self, function, function_uses_data=False):
        # lists easily function as stacks in python
        node_stack = [self.model.root]
        if function_uses_data:
            # keep track of each's node parent data after filtering
            data_stack = [self.data]
        
        filtered_data = [self.data]
        outputs = []

        while len(node_stack) > 0:
            node = node_stack.pop()

            if function_uses_data:
                data = data_stack.pop()

                # call function on filtered data from the node's parent
                outputs.append(function(node, filtered_data[-1]))

                filtered_data.append(self.filter_data_at_single_node(data, node))
            else:
                outputs.append(function(node))

            for child in self.model.children_dict[self.model.get_node_idx(node)]:
                node_stack.append(child)
                if function_uses_data:
                    # push the same data onto the stack since the
                    # children share the same parent
                    data_stack.append(filtered_data[-1])

                    # to generalize to arbitrary trees:
                    # move appending to filtered data here I think
                    # will work
        return outputs, filtered_data

    '''
    plots on an 1-d np array of axes the filtered data 
    and the gates for each node
    '''
    def plot_on_axes(self, axes, hparams):

        if not (axes.shape[0] == len(self.filtered_data) - 1):
            print(self.gates)
            print(len(self.filtered_data))
            raise ValueError('Number of axes must match the number of nodes!')

        for node_idx, axis in enumerate(axes):
            self.plot_node(axis, node_idx, hparams)

    # TODO refactor to use a dictionary of plot settings
    # which has a defautlt setting
    def plot_node(self, axis, node_idx, hparams):
        if 'plot_kde_density' in hparams['plot_params']:
            if hparams['plot_params']['plot_kde_density']:
                sb.kdeplot(
                    self.filtered_data[node_idx][:, self.dims[node_idx][0]],
                    self.filtered_data[node_idx][:, self.dims[node_idx][1]],
                    ax=axis, cmap='Blues', shade=True, shade_lowest=False
                )
        else:
            axis.scatter(
                self.filtered_data[node_idx][:, self.dims[node_idx][0]],
                self.filtered_data[node_idx][:, self.dims[node_idx][1]],
                s=hparams['plot_params']['marker_size'],
            )
#        if type(self.model.root).__name__ == 'ModelNode' or type(self.model.root).__name__ == 'SquareModelNode':
#            self.plot_gate(axis, node_idx, dashes=(3,1), label='Model')
#        else:
#            self.plot_gate(axis, node_idx, color='k', label='DAFI')
        if self.color is None:
            self.plot_gate(axis, node_idx, dashes=(3,1), label='Model')
        else:
            self.plot_gate(axis, node_idx, color=self.color, label='DAFI')


    def plot_only_DAFI_gates_on_axes(self, axes, hparams):
        if not (axes.shape[0] == len(self.filtered_data) - 1):
            raise ValueError('Number of axes must match the number of nodes!')
        for node_idx, axis in enumerate(axes):
            self.plot_gate(axis, node_idx, color='k', label='DAFI')



    def plot_gate(self, axis, node_idx, color='g', lw=3, dashes=(None, None), label=None):
        gate = self.gates[node_idx]
        axis.plot([gate.low1, gate.low1], [gate.low2, gate.upp2], c=color, 
            label=label, dashes=dashes, linewidth=lw)
        axis.plot([gate.low1, gate.upp1], [gate.low2, gate.low2], c=color, 
            dashes=dashes, linewidth=lw)
        axis.plot([gate.upp1, gate.upp1], [gate.low2, gate.upp2], c=color, 
            dashes=dashes, linewidth=lw)
        axis.plot([gate.upp1, gate.low1], [gate.upp2,gate.upp2], c=color, 
            dashes=dashes, linewidth=lw)
        return axis

    def plot_gate_return_line(self, axis, node_idx, color='g', lw=3, dashes=(None, None), label=None):
        gate = self.gates[node_idx]
        line, = axis.plot([gate.low1, gate.low1], [gate.low2, gate.upp2], c=color, 
            label=label, dashes=dashes, linewidth=lw)
        axis.plot([gate.low1, gate.upp1], [gate.low2, gate.low2], c=color, 
            dashes=dashes, linewidth=lw)
        axis.plot([gate.upp1, gate.upp1], [gate.low2, gate.upp2], c=color, 
            dashes=dashes, linewidth=lw)
        axis.plot([gate.upp1, gate.low1], [gate.upp2,gate.upp2], c=color, 
            dashes=dashes, linewidth=lw)
        return line


class DataAndGatesPlotterBoth(DataAndGatesPlotter):

    '''
    Class to handle plotting the data and gates

    attributes:
        model: the model whose gates to plot
        data: the data to plot
        filtered_data: data filtered along the tree structure 
                        defined by model in depth first order
        dims: the indexes into self.data for which two dimensions
                a given node uses
    '''
    def __init__(self, model, data_both, is_4chain=False, color=None):
        self.model = model
        self.data = data_both
        self.nodes = self.get_nodes()
        self.gates = self.construct_gates()
        self.filtered_data = self.filter_data_by_model_gates(data_both)
        self.dims = self.get_dims()
        self.color = color

        # modification needed to plot reference tree objects
        #ids2features = self.model.referenceTree.ids2features
        #self.feature_names = [(ids2features[dim1], ids2features[dim2]) 
        #        for (dim1, dim2) in self.dims]





    def get_dims(self):
        dims = []
        for node in self.nodes:
            dims.append([node.gate_dim1, node.gate_dim2])
        return dims
            
    def get_nodes(self):
        return self.model.get_flat_nodes()

    def construct_gates(self):
        gates = []
        for node in self.nodes:
           gates.append(ModelTree.get_gate(node))
        return gates



    def filter_data_by_model_gates(self, data_both): 
        return self.model.get_filtered_data_all_nodes(data_both)


    '''
    plots on an 1-d np array of axes the filtered data 
    and the gates for each node
    '''
    def plot_on_axes(self, axes, hparams):

        if not (axes.shape[0] == len(self.filtered_data)):
            print(self.gates)
            print(len(self.filtered_data))
            raise ValueError('Number of axes must match the number of nodes!')

        for node_idx, axis in enumerate(axes):
            self.plot_node(axis, node_idx, hparams)

    # TODO refactor to use a dictionary of plot settings
    # which has a defautlt setting
    def plot_node(self, axis, node_idx, hparams):
        if 'plot_kde_density' in hparams['plot_params']:
            if hparams['plot_params']['plot_kde_density']:
                sb.kdeplot(
                    self.filtered_data[node_idx][:, self.dims[node_idx][0]],
                    self.filtered_data[node_idx][:, self.dims[node_idx][1]],
                    ax=axis, cmap='Blues', shade=True, shade_lowest=False
                )
        else:
            axis.scatter(
                self.filtered_data[node_idx][:, self.dims[node_idx][0]],
                self.filtered_data[node_idx][:, self.dims[node_idx][1]],
                s=hparams['plot_params']['marker_size'],
            )
#        if type(self.model.root).__name__ == 'ModelNode' or type(self.model.root).__name__ == 'SquareModelNode':
#            self.plot_gate(axis, node_idx, dashes=(3,1), label='Model')
#        else:
#            self.plot_gate(axis, node_idx, color='k', label='DAFI')
        if self.color is None:
            self.plot_gate(axis, node_idx, dashes=(3,1), label='Model')
        else:
            self.plot_gate(axis, node_idx, color=self.color, label='DAFI')


    def plot_only_DAFI_gates_on_axes(self, axes, hparams):
        if not (axes.shape[0] == len(self.filtered_data) - 1):
            raise ValueError('Number of axes must match the number of nodes!')
        for node_idx, axis in enumerate(axes):
            self.plot_gate(axis, node_idx, color='k', label='DAFI')



    def plot_gate(self, axis, node_idx, color='g', lw=3, dashes=(None, None), label=None):
        gate = self.gates[node_idx]
        axis.plot([gate.low1, gate.low1], [gate.low2, gate.upp2], c=color, 
            label=label, dashes=dashes, linewidth=lw)
        axis.plot([gate.low1, gate.upp1], [gate.low2, gate.low2], c=color, 
            dashes=dashes, linewidth=lw)
        axis.plot([gate.upp1, gate.upp1], [gate.low2, gate.upp2], c=color, 
            dashes=dashes, linewidth=lw)
        axis.plot([gate.upp1, gate.low1], [gate.upp2,gate.upp2], c=color, 
            dashes=dashes, linewidth=lw)
        return axis

class DataAndGatesPlotterDepthOne(DataAndGatesPlotter):
    
    def __init__(self, model, data, color=None):

        self.model = model
        self.data = np.array(data)
        self.gates = [DepthOneModel.get_gate(node) for node in model.nodes]
        #self.filtered_data = [self.data for d in range(len(model.nodes) + 1)]
        self.dims = np.array([[0, 1] for d in range(len(model.nodes))])
        self.color = color
    
    @staticmethod
    def get_per_cell_labels(cell_data, labels):
        return np.array(np.concatenate([labels[i] * torch.ones([cell_data[i].shape[0], 1]) for i in range(len(cell_data))]))

    def plot_data_with_gates(self, cell_labels, size='default', figscale=8):
        data_pos = self.data[cell_labels[:, 0] == 1]
        data_neg = self.data[cell_labels[:, 0] == 0]
        if size == 'default':
            # just a heuristic to get a decent marker size
            size = 1000 * 1/self.data.shape[0]
        else:
            size = size
        fig, axes = plt.subplots(2, 1, figsize=(figscale, figscale * 2))
        axes[0].scatter(data_pos[:, 0], data_pos[:, 1], color='r', s=size)
        axes[1].scatter(data_neg[:, 0], data_neg[:, 1], color='b', s=size)
        lines_for_legend = self.plot_all_gates(axes[0]) # -1 was here before???
        pos_feats = self.model(torch.tensor(data_pos.reshape([1, data_pos.shape[0],  -1])), torch.ones([1]))['leaf_logp'][0, :]
        neg_feats = self.model(torch.tensor(data_neg.reshape([1, data_neg.shape[0], -1])), torch.zeros([1]))['leaf_logp'][0, :]
                

        self.plot_all_gates(axes[1])
        
        legend1 = axes[0].legend(loc='upper right')
        legend2 = axes[0].legend([line for line in lines_for_legend], [str(pos_feat.cpu().detach().numpy()) for pos_feat in pos_feats], loc='lower left')
        axes[0].add_artist(legend1)
        axes[1].legend([line for line in lines_for_legend], [str(neg_feat.cpu().detach().numpy()) for neg_feat in neg_feats], loc='lower left')


          
    def plot_all_gates(self, axis):
        cm = plt.get_cmap('hsv')
        colors = cm(np.linspace(0, 1, len(self.gates)))
        lines_for_legend = []
        for g in range(len(self.gates)):
            line = self.plot_gate_return_line(axis, g, color=colors[g], label='%.4f' %self.model.linear.weight[0].cpu().detach().numpy()[g])
            lines_for_legend.append(line)
        return lines_for_legend
            

    def plot_inverse_UMAP_transform_in_feature_space(self, umapper, untransformed_data, gate_data_idxs=None, figlen=10, ms=.1):
        matplotlib.rcParams.update({'font.size': 22})
        data_inside_first_gate_idxs = self.filter_data_at_single_node(self.data, self.model.nodes[0], return_idxs=True)
        
        data_inside_first_gate_inverse_transform = untransformed_data[data_inside_first_gate_idxs]#umapper.inverse_transform(data_inside_first_gate)

        # plot gates instead of arbitrary pairs of data
        if gate_data_idxs:
        # 2,3 -> SSC-H, CD45 
        # 0, 1 -> FSC-A, SSC-A
        # 5, 6 -> CD5, CD19
            gate_names = {(2, 3): 'SSC-H CD45', (0, 1): 'FSC-A SSC-A', (5, 6): 'CD5 CD19', (10, 7): 'CD10 CD79b'}
            fig, axes = plt.subplots(len(gate_data_idxs), 1, figsize=(figlen * 1, figlen * len(gate_data_idxs)))
            for axis, gate_idxs in zip(axes, gate_data_idxs):
                axis.scatter(untransformed_data[:, gate_idxs[0]], untransformed_data[:, gate_idxs[1]], c='lightgrey', s=ms/10)
                axis.scatter(data_inside_first_gate_inverse_transform[:, gate_idxs[0]], data_inside_first_gate_inverse_transform[:, gate_idxs[1]], c='r', s=ms)
                axis.set_title(gate_names[tuple(gate_idxs)])
        # plot arbitrary pairs of data
        else:
            fig, axes = plt.subplots(int(data_inside_first_gate_inverse_transform.shape[1]/2), 1, figsize=(figlen * 1, figlen * int(data_inside_first_gate_inverse_transform.shape[1]/2)))
            for i, axis in zip(range(data_inside_first_gate_inverse_transform.shape[1]), axes):
                axis.scatter(untransformed_data[:, [2 * i]], untransformed_data[:, [2 * i + 1]], c='lightgrey', s=ms/10)
                axis.scatter(data_inside_first_gate_inverse_transform[:, [2 * i]], data_inside_first_gate_inverse_transform[:, [2 * i + 1]], c='r', s=ms)

                
                
    def plot_inverse_UMAP_transform_in_feature_space_with_filtering(self, umapper, untransformed_data, gate_data_idxs=None, figlen=5, ms=.1):
        data_inside_first_gate_idxs = self.filter_data_at_single_node(self.data, self.model.nodes[0], return_idxs=True)
        
        data_inside_first_gate_inverse_transform = untransformed_data[data_inside_first_gate_idxs]#umapper.inverse_transform(data_inside_first_gate)

        fig, axes = plt.subplots(len(gate_data_idxs), 1, figsize=(figlen * 1, figlen * len(gate_data_idxs)))
        #cur_untransformed_data = self.filter_root_untransformed_cll(cur_untransformed_data)
        #cur_data_inside_first_gate_inverse_transform = self.filter_root_untransformed_cll(cur_data_inside_first_gate_inverse_transform)
        cur_untransformed_data = untransformed_data
        cur_data_inside_first_gate_inverse_transform = data_inside_first_gate_inverse_transform
        i = 0
        for axis, gate_idxs in zip(axes, gate_data_idxs):
            if not(i == 0):
                cur_untransformed_data = self.filter_untransformed_cll(gate_data_idxs[i - 1], cur_untransformed_data)
                cur_data_inside_first_gate_inverse_transform = self.filter_untransformed_cll(gate_data_idxs[i - 1], cur_data_inside_first_gate_inverse_transform)
            axis.scatter(cur_untransformed_data[:, gate_idxs[0]], cur_untransformed_data[:, gate_idxs[1]], c='lightgrey', s=ms/10)
            axis.scatter(cur_data_inside_first_gate_inverse_transform[:, gate_idxs[0]], cur_data_inside_first_gate_inverse_transform[:, gate_idxs[1]], c='r', s=ms)
            i += 1

    #def filter_root_untransformed_cll(self, data):
    #    root_gate_data_idxs = []
    #    root_gate_slope = blah
    #    dh.blah
    
    def filter_untransformed_cll(self, gate_idxs, data):
        # 2,3 -> SSC-H, CD45 
        # 0, 1 -> FSC-A, SSC-A
        # 5, 6 -> CD5, CD19
        gate_boundaries_matching_data = {(2, 3): [102, 921, 2048, 3891], (0, 1): [921, 2150, 102, 921], (5, 6):[1638, 3891, 2150, 3891]}
        gate = gate_boundaries_matching_data[tuple(gate_idxs)]
        filtered_data = dh.filter_rectangle(
                data, gate_idxs[0], 
                gate_idxs[1], gate[0], gate[1], 
                gate[2], gate[3],
        )
        return filtered_data
        

