import torch
import math
import numpy as np
from collections import namedtuple
import utils.utils_load_data as dh
import torch.nn.functional as F
from utils.DepthOneModel import DepthOneModel
from utils.bayes_gate import ModelTree
import seaborn as sb
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.mplot3d import Axes3D

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
        self.dims = self.get_dims()
        self.filtered_data = self.filter_data_by_model_gates()
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
        if type(node).__name__ == 'SquareModelNode':
            return_value = dh.filter_rectangle(
                    data, node.gate_dim1, 
                    node.gate_dim2, gate.low1, gate.upp1, 
                    gate.low2, gate.upp2,
                    return_idx=return_idxs
            )
        elif type(node).__name__ == 'SphericalModelNode':
            gate[0] = [center.item() for center in gate[0]]
            gate[1] = gate[1].item()
            dist = ( 
                (data[:, 0] - gate[0][0])**2 +\
                (data[:, 1] - gate[0][1])**2 +\
                (data[:, 2] - gate[0][2])**2\
            )**(1/2)

            return_value = dist <= gate[1]
        elif type(node).__name__ == 'EllipticalModelNode':
            ell_gate = node.get_gate()
            center = ell_gate[0]
            a = ell_gate[1]
            b = ell_gate[2]
            theta = ell_gate[3]
            dist = node.compute_dist_to_ellipse(
                center[0], center[1], a, b, theta, torch.tensor(data)
            )
            return_value = dist <= 0
            return_value = torch.tensor(np.array([True if ret == 1 else False for ret in return_value])    )
            print(return_value)
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
            #print(self.gates)
            #print(len(self.filtered_data))
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
        if self.model.node_type == 'circular':
            self.plot_gate_circular(axis, node_idx, color=color, lw=lw, dashes=dashes, label=label)
        elif self.model.node_type == 'axis_aligned_elliptical':
            self.plot_gate_axis_aligned_elliptical(axis, node_idx, color=color, lw=lw, dashes=dashes, label=label)
        elif self.model.node_type == 'elliptical':
            self.plot_gate_elliptical(axis, node_idx, color=color, lw=lw, dashes=dashes, label=label)
        elif self.model.node_type == 'spherical':
            self.plot_gate_spherical(axis, node_idx, color=color, lw=lw, dashes=dashes, label=label)
        else:
            self.plot_gate_rectangular(axis, node_idx, color=color, lw=lw, dashes=dashes, label=label)

    def plot_gate_return_line(self, axis, node_idx, color='g', lw=3, dashes=(None, None), label=None):
        if self.model.node_type == 'circular':
            line = self.plot_gate_return_line_circular(axis, node_idx, color=color, lw=lw, dashes=dashes, label=label)
        elif self.model.node_type == 'axis_aligned_elliptical':
            line = self.plot_gate_return_line_axis_aligned_elliptical(axis, node_idx, color=color, lw=lw, dashes=dashes, label=label)
        elif self.model.node_type == 'elliptical':
            line = self.plot_gate_return_line_elliptical(axis, node_idx, color=color, lw=lw, dashes=dashes, label=label)
        elif self.model.node_type == 'spherical':
            line = self.plot_gate_return_line_spherical(axis, node_idx, color=color, lw=lw, dashes=dashes, label=label)
        elif self.model.node_type == 'ball':
            line = self.plot_gate_return_line_ball(axis, node_idx, color=color, lw=lw, dashes=dashes, label=label)
        else:
            line = self.plot_gate_return_line_rectangular(axis, node_idx, color=color, lw=lw, dashes=dashes, label=label)
        return line

    def plot_gate_circular(self, axis, node_idx, color='g', lw=3, dashes=(None, None), label=None):
        gate = self.gates[node_idx]
        print(gate)
        circle = plt.Circle(gate[0], gate[1], color=color, fill=False)
        axis.add_artist(circle)

    def plot_gate_axis_aligned_elliptical(self, axis, node_idx, color='g', lw=3, dashes=(None, None), label=None):
        gate = self.gates[node_idx]
        
        center1 = gate[0][0].item()
        center2 = gate[0][1].item()
        a = gate[1].item()
        b = gate[2].item()

        print(gate)

        theta = np.linspace(0, 2*math.pi, 150)
        plt.plot(center1 + a * np.cos(theta) , center2 + b*np.sin(theta))



    def plot_gate_elliptical(self, axis, node_idx, color='g', lw=3, dashes=(None, None), label=None):
        gate = self.gates[node_idx]
        
        center1 = gate[0][0].item()
        center2 = gate[0][1].item()
        a = gate[1].item()
        b = gate[2].item()
        theta = gate[3].item()

        print(gate)

        phi = np.linspace(0, 2*math.pi, 150)
        plt.plot(
            center1 + a * np.cos(theta) * np.cos(phi) - b * np.sin(theta) * np.sin(phi), 
            center2 + a * np.cos(phi) * np.sin(theta) + b * np.sin(phi) * np.cos(theta)
        )

    def plot_gate_spherical(self, axis, node_idx, color='g', lw=3, dashes=(None, None), label=None):
        gate = self.gates[node_idx]
        center1 = gate[0][0].item()
        center2 = gate[0][1].item()
        center3 = gate[0][2].item()
        radius = gate[1].item()

        theta = np.linspace(0, 2 * np.pi, 100)
        phi = np.linspace(0, np.pi, 100)
    
        x = radius * np.outer(np.cos(theta), np.sin(phi)) + center1
        y = radius * np.outer(np.sin(theta), np.sin(phi)) + center2
        z = radius * np.outer(np.ones(np.size(theta)), np.cos(phi)) + center3

        axis.plot_surface(x, y, z, alpha=.3, color='r')

        
    def plot_gate_return_line_circular(self, axis, node_idx, color='g', lw=3, dashes=(None, None), label=None):
        gate = self.gates[node_idx]
        circle = plt.Circle(gate[0], gate[1], color=color, fill=False)
        axis.add_artist(circle)
        return circle

    def plot_gate_return_line_ball(self, axis, node_idx, color='g', lw=3, dashes=(None, None), label=None):
        gate = self.gates[node_idx]
        gate[0] = gate[0] * self.old_scale + self.old_offset;
        center = self.transformer.transform([gate[0]])[0]
        print(center)
        circle = plt.Circle(center, gate[1], color=color, fill=False)
        axis.add_artist(circle)
        return circle

    def plot_gate_return_line_axis_aligned_elliptical(self, axis, node_idx, color='g', lw=3, dashes=(None, None), label=None):
        gate = self.gates[node_idx]
        center1 = gate[0][0].item()
        center2 = gate[0][1].item()
        a = gate[1].item()
        b = gate[2].item()
        print(gate)

        theta = np.linspace(0, 2*math.pi, 150)
        line, = axis.plot(center1 + a * np.cos(theta) , center2 + b*np.sin(theta))
        return line

    def plot_gate_return_line_elliptical(self, axis, node_idx, color='g', lw=3, dashes=(None, None), label=None):
        gate = self.gates[node_idx]
        
        center1 = gate[0][0].item()
        center2 = gate[0][1].item()
        a = gate[1].item()
        b = gate[2].item()
        theta = gate[3].item()

        print(gate)

        phi = np.linspace(0, 2*math.pi, 150)
        line, = axis.plot(
            center1 + a * np.cos(theta) * np.cos(phi) - b * np.sin(theta) * np.sin(phi), 
            center2 + a * np.cos(phi) * np.sin(theta) + b * np.sin(phi) * np.cos(theta)
        )
        return line

    def plot_gate_return_line_spherical(self, axis, node_idx, color='g', lw=3, dashes=(None, None), label=None):
        gate = self.gates[node_idx]
        center1 = gate[0][0].item()
        center2 = gate[0][1].item()
        center3 = gate[0][2].item()
        radius = gate[1].item()

        theta = np.linspace(0, 2 * np.pi, 100)
        phi = np.linspace(0, np.pi, 100)
    
        x = radius * np.outer(np.cos(theta), np.sin(phi)) + center1
        y = radius * np.outer(np.sin(theta), np.sin(phi)) + center2
        z = radius * np.outer(np.ones(np.size(theta)), np.cos(phi)) + center3

        out = axis.plot_surface(x, y, z, alpha=.3, color='r')

        return out

    def plot_gate_rectangular(self, axis, node_idx, color='g', lw=3, dashes=(None, None), label=None):
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

    def plot_gate_return_line_rectangular(self, axis, node_idx, color='g', lw=3, dashes=(None, None), label=None):
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
        if self.data.shape[1] == 2:
            fig, axes = self.plot_data_with_gates_2d(cell_labels, size=size, figscale=figscale)
            return fig, axes
        else:
            fig_pos, ax_pos, fig_neg, ax_neg = self.plot_data_with_gates_3d(cell_labels, size=size, figscale=figscale)
            return fig_pos, ax_pos, fig_neg, ax_neg

    def plot_data_with_gates_2d(self, cell_labels, size='default', figscale=8):
        data_pos = self.data[cell_labels[:, 0] == 1]
        data_neg = self.data[cell_labels[:, 0] == 0]
        if size == 'default':
            # just a heuristic to get a decent marker size
            size = 1000 * 1/self.data.shape[0]
        else:
            size = size
        fig, axes = plt.subplots(2, 1, figsize=(figscale, figscale * 2))
        print(data_pos.shape, size, data_neg.shape, 'data pos shape, size, data_neg shape')
        axes[0].scatter(data_pos[:, 0], data_pos[:, 1], color='k', s=size)
        axes[1].scatter(data_neg[:, 0], data_neg[:, 1], color='k', s=size)
        axes[0].set_title('Positive Cells in UMAP Space')
        axes[1].set_title('Negative Cells in UMAP Space')
        lines_for_legend = self.plot_all_gates(axes[0]) # -1 was here before???
        pos_feats = self.model(torch.tensor(data_pos.reshape([1, data_pos.shape[0],  -1])), torch.ones([1]))['leaf_logp'][0, :]
        neg_feats = self.model(torch.tensor(data_neg.reshape([1, data_neg.shape[0], -1])), torch.zeros([1]))['leaf_logp'][0, :]
                

        self.plot_all_gates(axes[1])
        
        legend1 = axes[0].legend(loc='upper right')
        legend2 = axes[0].legend([line for line in lines_for_legend], [str(pos_feat.cpu().detach().numpy()) for pos_feat in pos_feats], loc='lower left', title='Average Pos Feature')
        #axes[0].add_artist(legend1)
        axes[1].legend([line for line in lines_for_legend], [str(neg_feat.cpu().detach().numpy()) for neg_feat in neg_feats], loc='lower left', title='Average Neg Feature')
#        axes[0].legend.title('Average Pos Feature')
#        axes[1].legend.title('Average Neg Feature')
        return fig, axes

    def plot_data_with_gates_3d(self, cell_labels, size='default', figscale=8):
        data_pos = self.data[cell_labels[:, 0] == 1]
        data_neg = self.data[cell_labels[:, 0] == 0]
        if size == 'default':
            # just a heuristic to get a decent marker size
            size = 1000 * 1/self.data.shape[0]
        else:
            size = size
        _rawfig_pos = plt.figure()
        ax_pos = fig_pos.add_subplot(111, projection='3d')
        ax_pos.scatter(data_pos[:, 0], data_pos[:, 1], data_pos[:, 2])

        fig_neg = plt.figure()
        ax_neg = fig_neg.add_subplot(111, projection='3d')
        ax_neg.scatter(data_neg[:, 0], data_neg[:, 1], data_neg[:, 2])


        self.plot_all_gates(ax_pos) # -1 was here before???
        #pos_feats = self.model(torch.tensor(data_pos.reshape([1, data_pos.shape[0],  -1])), torch.ones([1]))['leaf_logp'][0, :]
        #neg_feats = self.model(torch.tensor(data_neg.reshape([1, data_neg.shape[0], -1])), torch.zeros([1]))['leaf_logp'][0, :]
                

        self.plot_all_gates(ax_neg)
        
        #legend1 = axes[0].legend(loc='upper right')
        #legend2 = axes[0].legend([line for line in lines_for_legend], [str(pos_feat.cpu().detach().numpy()) for pos_feat in pos_feats], loc='lower left', title='Average Pos Feature')
        #axes[0].add_artist(legend1)
        #axes[1].legend([line for line in lines_for_legend], [str(neg_feat.cpu().detach().numpy()) for neg_feat in neg_feats], loc='lower left', title='Average Neg Feature')
#        axes[0].legend.title('Average Pos Feature')
#        axes[1].legend.title('Average Neg Feature')
        return fig_pos, ax_pos, fig_neg, ax_neg


    def plot_single_sample_with_gate(self, 
        sample, sample_id, true_label, axis,
        size='default', color='k', include_diagnostics=True,
        true_feature=None, BALL=False
    ):
        if size == 'default':
            size = 1000 * 1/self.data.shape[0]
        else:
            size = size
        axis.scatter(sample[:, 0], sample[:, 1], color=color, s=size)
        if not BALL:
            axis.text(0.02, .02, '%d' %sample_id)
        else:
            axis.text(0.80, .02, '%d' %sample_id)
            
        lines_for_legend = self.plot_all_gates(axis)
        output = self.model([torch.tensor(sample, dtype=torch.float)], torch.tensor([true_label], dtype=torch.float))
        feats = torch.exp(output['leaf_logp'][0]) * 100
        pred = (output['y_pred'].cpu().detach().numpy() >= .5) * 1.0
        prob = output['y_pred']
        
        if include_diagnostics:
            diagnostics_str = ''
            if not (true_feature is None):
                diagnostics_str += 'True Feature: %.2f' % true_feature + '\n'
            diagnostics_str += 'Feature: %.2f' %feats
            diagnostics_str += '\nPredicted Label: %d' %pred
            diagnostics_str += '\nTrue Label: %d' %true_label
            diagnostics_str += '\nPr(y=1|x): %.2f' %prob
            if not BALL:
                axis.text(.55, .02, diagnostics_str)
            else:
                axis.text(0.02, .02, diagnostics_str)
            
        #axis.legend([line for line in lines_for_legend], [str(feat.cpu().detach().numpy()) for feat in feats], loc='lower left')
        
        
        
          
    def plot_all_gates(self, axis):
        cm = plt.get_cmap('hsv')
        colors = cm(np.linspace(0, 1, len(self.gates)))
        lines_for_legend = []
        for g in range(len(self.gates)):
            line = self.plot_gate_return_line(axis, g, color=colors[g], label='%.4f' %self.model.linear.weight[0].cpu().detach().numpy()[g])
            lines_for_legend.append(line)
        return lines_for_legend
            

    def plot_inverse_UMAP_transform_in_feature_space(self, umapper, untransformed_data, gate_data_idxs=None, figlen=10, ms=.1, BALL=False):
        matplotlib.rcParams.update({'font.size': 22})
        data_inside_first_gate_idxs = self.filter_data_at_single_node(self.data, self.model.nodes[0], return_idxs=True)
        
        data_inside_first_gate_inverse_transform = untransformed_data[data_inside_first_gate_idxs]#umapper.inverse_transform(data_inside_first_gate)

        # plot gates instead of arbitrary pairs of data
        if gate_data_idxs:
            if not BALL:
                gate_names = {(3, 4): 'SSC-H CD45', (0, 2): 'FSC-A SSC-A', (6, 7): 'CD5 CD19', (11, 8): 'CD10 CD79b', (0, 1):'FSC-A FSC-H', (2, 3): 'SSC-A SSC-H', (-1, 5): 'CD 38 CD22'}
            else:
                gate_names = {
                    (0, 1): 'FSC SSC',
                    (2, 5):'CD66b CD24',
                    (9, 6): 'CD20 CD10',
                    (7, 8): 'CD34 CD38',
                    (4, 1): 'CD19 SSC',
                    (7, 10): 'CD34 CD45',
                    (6, 10): 'CD10 CD45',
                    (7, 3): 'CD34 CD22',
                }
                gate_data_idxs = [[0, 1], [2, 5], [9, 6], [7, 8], [4, 1], [7, 10], [6, 10], [7, 3]]
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


class MultidimDataAndGatesPlotter(DataAndGatesPlotter):
    def __init__(self, model, data, data_raw, old_scale, old_offset, transformer, color=None):
        self.model = model
        self.transformer = transformer
        self.data = np.array(data)
        self.data_raw = np.array(data_raw)
        self.old_scale = old_scale
        self.old_offset = old_offset
        self.gates = [DepthOneModel.get_gate(node) for node in model.nodes]
        #self.filtered_data = [self.data for d in range(len(model.nodes) + 1)]
        self.dims = np.array([[0, 1] for d in range(len(model.nodes))])
        self.color = color

    @staticmethod
    def get_per_cell_labels(cell_data, labels):
        return np.array(np.concatenate([labels[i] * torch.ones([cell_data[i].shape[0], 1]) for i in range(len(cell_data))]))

    def plot_data_with_gates(self, cell_labels, size='default', figscale=8):
        if self.data.shape[1] == 2:
            fig, axes = self.plot_data_with_gates_2d(cell_labels, size=size, figscale=figscale)
            return fig, axes
        else:
            fig_pos, ax_pos, fig_neg, ax_neg = self.plot_data_with_gates_3d(cell_labels, size=size, figscale=figscale)
            return fig_pos, ax_pos, fig_neg, ax_neg

    def plot_data_with_gates_2d(self, cell_labels, size='default', figscale=8):
        data_pos = self.data[cell_labels[:, 0] == 1]
        data_neg = self.data[cell_labels[:, 0] == 0]
        raw_pos = self.data_raw[cell_labels[:, 0] == 1]
        raw_neg = self.data_raw[cell_labels[:, 0] == 0]
        if size == 'default':
            # just a heuristic to get a decent marker size
            size = 1000 * 1/self.data.shape[0]
        else:
            size = size
        fig, axes = plt.subplots(2, 1, figsize=(figscale, figscale * 2))
        print(data_pos.shape, size, data_neg.shape, 'data pos shape, size, data_neg shape')

        # find which points are in the gate
        gate = self.gates[0]
        center = np.concatenate([g.detach().numpy() for g in gate[0]]) #  * self.old_scale + self.old_offset;
        radius = gate[1].detach().numpy() # * self.old_scale

        pos_in_gate = np.linalg.norm((raw_pos - self.old_offset)/self.old_scale - center, axis=1) < radius
        neg_in_gate = np.linalg.norm((raw_neg - self.old_offset)/self.old_scale - center, axis=1) < radius

        print(center)
        print(radius)
        print(raw_pos[0,:])
        print(np.any(pos_in_gate))

        axes[0].scatter(data_pos[~pos_in_gate][:, 0], data_pos[~pos_in_gate][:, 1], color='k', s=size)
        axes[0].scatter(data_pos[pos_in_gate][:, 0], data_pos[pos_in_gate][:, 1], color='r', s=size)
        axes[1].scatter(data_neg[~neg_in_gate][:, 0], data_neg[~neg_in_gate][:, 1], color='k', s=size)
        axes[1].scatter(data_neg[neg_in_gate][:, 0], data_neg[neg_in_gate][:, 1], color='r', s=size)
        axes[0].set_title('Positive Cells in UMAP Space')
        axes[1].set_title('Negative Cells in UMAP Space')
        ## lines_for_legend = self.plot_all_gates(axes[0]) # -1 was here before???
        ## pos_feats = self.model(torch.tensor(data_pos.reshape([1, data_pos.shape[0],  -1])), torch.ones([1]))['leaf_logp'][0, :]
        ## neg_feats = self.model(torch.tensor(data_neg.reshape([1, data_neg.shape[0], -1])), torch.zeros([1]))['leaf_logp'][0, :]
                

        ## self.plot_all_gates(axes[1])
        
        ## legend1 = axes[0].legend(loc='upper right')
        ## legend2 = axes[0].legend([line for line in lines_for_legend], [str(pos_feat.cpu().detach().numpy()) for pos_feat in pos_feats], loc='lower left', title='Average Pos Feature')
        #axes[0].add_artist(legend1)
        ##axes[1].legend([line for line in lines_for_legend], [str(neg_feat.cpu().detach().numpy()) for neg_feat in neg_feats], loc='lower left', title='Average Neg Feature')
#        axes[0].legend.title('Average Pos Feature')
#        axes[1].legend.title('Average Neg Feature')
        return fig, axes

    def plot_data_with_gates_3d(self, cell_labels, size='default', figscale=8):
        data_pos = self.data[cell_labels[:, 0] == 1]
        data_neg = self.data[cell_labels[:, 0] == 0]
        if size == 'default':
            # just a heuristic to get a decent marker size
            size = 1000 * 1/self.data.shape[0]
        else:
            size = size
        fig_pos = plt.figure()
        ax_pos = fig_pos.add_subplot(111, projection='3d')
        ax_pos.scatter(data_pos[:, 0], data_pos[:, 1], data_pos[:, 2])

        fig_neg = plt.figure()
        ax_neg = fig_neg.add_subplot(111, projection='3d')
        ax_neg.scatter(data_neg[:, 0], data_neg[:, 1], data_neg[:, 2])


        self.plot_all_gates(ax_pos) # -1 was here before???
        #pos_feats = self.model(torch.tensor(data_pos.reshape([1, data_pos.shape[0],  -1])), torch.ones([1]))['leaf_logp'][0, :]
        #neg_feats = self.model(torch.tensor(data_neg.reshape([1, data_neg.shape[0], -1])), torch.zeros([1]))['leaf_logp'][0, :]
                

        self.plot_all_gates(ax_neg)
        
        #legend1 = axes[0].legend(loc='upper right')
        #legend2 = axes[0].legend([line for line in lines_for_legend], [str(pos_feat.cpu().detach().numpy()) for pos_feat in pos_feats], loc='lower left', title='Average Pos Feature')
        #axes[0].add_artist(legend1)
        #axes[1].legend([line for line in lines_for_legend], [str(neg_feat.cpu().detach().numpy()) for neg_feat in neg_feats], loc='lower left', title='Average Neg Feature')
#        axes[0].legend.title('Average Pos Feature')
#        axes[1].legend.title('Average Neg Feature')
        return fig_pos, ax_pos, fig_neg, ax_neg


    def plot_single_sample_with_gate(self, 
        sample, sample_id, true_label, axis,
        size='default', color='k', include_diagnostics=True,
        true_feature=None, BALL=False
    ):
        if size == 'default':
            size = 1000 * 1/self.data.shape[0]
        else:
            size = size
        axis.scatter(sample[:, 0], sample[:, 1], color=color, s=size)
        if not BALL:
            axis.text(0.02, .02, '%d' %sample_id)
        else:
            axis.text(0.80, .02, '%d' %sample_id)
            
        lines_for_legend = self.plot_all_gates(axis)
        output = self.model([torch.tensor(sample, dtype=torch.float)], torch.tensor([true_label], dtype=torch.float))
        feats = torch.exp(output['leaf_logp'][0]) * 100
        pred = (output['y_pred'].cpu().detach().numpy() >= .5) * 1.0
        prob = output['y_pred']
        
        if include_diagnostics:
            diagnostics_str = ''
            if not (true_feature is None):
                diagnostics_str += 'True Feature: %.2f' % true_feature + '\n'
            diagnostics_str += 'Feature: %.2f' %feats
            diagnostics_str += '\nPredicted Label: %d' %pred
            diagnostics_str += '\nTrue Label: %d' %true_label
            diagnostics_str += '\nPr(y=1|x): %.2f' %prob
            if not BALL:
                axis.text(.55, .02, diagnostics_str)
            else:
                axis.text(0.02, .02, diagnostics_str)
            
        #axis.legend([line for line in lines_for_legend], [str(feat.cpu().detach().numpy()) for feat in feats], loc='lower left')
        
        
        
          
    def plot_all_gates(self, axis):
        cm = plt.get_cmap('hsv')
        colors = cm(np.linspace(0, 1, len(self.gates)))
        lines_for_legend = []
        for g in range(len(self.gates)):
            line = self.plot_gate_return_line(axis, g, color=colors[g], label='%.4f' %self.model.linear.weight[0].cpu().detach().numpy()[g])
            lines_for_legend.append(line)
        return lines_for_legend
            

    def plot_inverse_UMAP_transform_in_feature_space(self, untransformed_data, gate_data_idxs=None, figlen=10, ms=.1, BALL=False):
        matplotlib.rcParams.update({'font.size': 22})
        data_inside_first_gate_idxs = self.filter_data_at_single_node(self.data, self.model.nodes[0], return_idxs=True)
        
        data_inside_first_gate_inverse_transform = untransformed_data[data_inside_first_gate_idxs]#umapper.inverse_transform(data_inside_first_gate)

        # plot gates instead of arbitrary pairs of data
        if gate_data_idxs:
            if not BALL:
                gate_names = {(3, 4): 'SSC-H CD45', (0, 2): 'FSC-A SSC-A', (6, 7): 'CD5 CD19', (11, 8): 'CD10 CD79b', (0, 1):'FSC-A FSC-H', (2, 3): 'SSC-A SSC-H', (-1, 5): 'CD 38 CD22'}
            else:
                gate_names = {
                    (0, 1): 'FSC SSC',
                    (2, 5):'CD66b CD24',
                    (9, 6): 'CD20 CD10',
                    (7, 8): 'CD34 CD38',
                    (4, 1): 'CD19 SSC',
                    (7, 10): 'CD34 CD45',
                    (6, 10): 'CD10 CD45',
                    (7, 3): 'CD34 CD22',
                }
                gate_data_idxs = [[0, 1], [2, 5], [9, 6], [7, 8], [4, 1], [7, 10], [6, 10], [7, 3]]
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

    def plot_in_feature_space(self, cell_labels, size=None, figlen=5):
        data_pos = self.data[cell_labels[:, 0] == 1]
        data_neg = self.data[cell_labels[:, 0] == 0]
        raw_pos = self.data_raw[cell_labels[:, 0] == 1]
        raw_neg = self.data_raw[cell_labels[:, 0] == 0]

        # find which points are in the gate
        gate = self.gates[0]
        center = np.concatenate([g.detach().numpy() for g in gate[0]]) #  * self.old_scale + self.old_offset;
        radius = gate[1].detach().numpy() # * self.old_scale

        pos_in_gate = np.linalg.norm((raw_pos - self.old_offset)/self.old_scale - center, axis=1) < radius
        neg_in_gate = np.linalg.norm((raw_neg - self.old_offset)/self.old_scale - center, axis=1) < radius

        if size == None:
            # just a heuristic to get a decent marker size
            size = 1000 * 1/self.data.shape[0]
        else:
            size = size

        gate_names = {
            (3, 4): 'SSC-H CD45',
            (0, 2): 'FSC-A SSC-A',
            (6, 7): 'CD5 CD19',
            (11, 8): 'CD10 CD79b',
            (0, 1): 'FSC-A FSC-H',
            (2, 3): 'SSC-A SSC-H',
            (-1, 5): 'CD 38 CD22'
        }
        gate_data_idxs = [[3, 4], [0, 2], [6, 7], [11, 8], [0, 1], [2, 3], [-1, 5]]
        fig, axes = plt.subplots(len(gate_data_idxs), 2, figsize=(figlen * 2, figlen * len(gate_data_idxs)))
        for axs, gate_idxs in zip(axes, gate_data_idxs):
            axs[0].scatter(raw_pos[~pos_in_gate][:, gate_idxs[0]], raw_pos[~pos_in_gate][:, gate_idxs[1]], color='#AAAAAA', s=size)
            axs[0].scatter(raw_pos[pos_in_gate][:, gate_idxs[0]], raw_pos[pos_in_gate][:, gate_idxs[1]], color='r', s=size)
            axs[1].scatter(raw_neg[~neg_in_gate][:, gate_idxs[0]], raw_neg[~neg_in_gate][:, gate_idxs[1]], color='#AAAAAA', s=size)
            axs[1].scatter(raw_neg[neg_in_gate][:, gate_idxs[0]], raw_neg[neg_in_gate][:, gate_idxs[1]], color='r', s=size)
            axs[0].set_title('Positive Cells in %s Space' % gate_names[tuple(gate_idxs)])
            axs[1].set_title('Negative Cells in %s Space' % gate_names[tuple(gate_idxs)])

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
