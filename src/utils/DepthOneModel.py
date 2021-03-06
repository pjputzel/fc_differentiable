from utils.bayes_gate import ModelTree
from utils.bayes_gate import ModelNode
from utils.bayes_gate import SquareModelNode
from utils.bayes_gate import CircularModelNode
from utils.bayes_gate import AxisAlignedEllipticalModelNode
from utils.bayes_gate import EllipticalModelNode
from utils.bayes_gate import SphericalModelNode
from utils.bayes_gate import BallModelNode
from utils.bayes_gate import HyperrectangleModelNode
from utils.bayes_gate import Gate
import torch 
import torch.nn as nn
import numpy as np

# simple class to make this depth one model work with the older # model node code
class InitGate:
    def __init__(self, gate):
#        print(gate)
        self.gate = Gate(gate, {'D1': 0, 'D2':1})

class DepthOneModel(ModelTree):
    def __init__(self, init_gates, model_params):
        super(ModelTree, self).__init__()
        for key in model_params:
            setattr(self, key, model_params[key])
        self.linear = nn.Linear(len(init_gates), 1)
        self.criterion = nn.BCEWithLogitsLoss()
        self.init_nodes(init_gates)
        self.num_gates = len(self.nodes)

    def init_nodes(self, init_gates):
        self.nodes = nn.ModuleList()
        for gate in init_gates:
            self.nodes.append(self.get_node(gate))

    def add_node(self, gate):
        self.nodes.append(self.get_node(gate))
        with torch.no_grad():
            self.linear.weight = nn.Parameter(torch.cat((self.linear.weight, torch.randn(1, 1)), dim=1))
#            self.linear.bias = nn.Parameter(torch.cat((self.linear.bias, torch.randn(1))))
        self.num_gates = self.num_gates + 1
        
    def get_node(self, gate):
        print(self.logistic_k)
        if self.node_type == 'square':
            gate = InitGate(gate) 
            node = SquareModelNode(self.logistic_k, gate, gate_size_default=self.gate_size_default)
        elif self.node_type == 'rectangle':
            gate = InitGate(gate) 
            node = ModelNode(self.logistic_k, gate, gate_size_default=self.gate_size_default)
        elif self.node_type == 'circular':
            node = CircularModelNode(self.logistic_k, gate, gate_size_default=self.gate_size_default, gate_dim1='D1', gate_dim2='D2')
        elif self.node_type == 'axis_aligned_elliptical':
            node = AxisAlignedEllipticalModelNode(self.logistic_k, gate, gate_size_default=self.gate_size_default, gate_dim1='D1', gate_dim2='D2')

        elif self.node_type == 'elliptical':
            node = EllipticalModelNode(self.logistic_k, gate, gate_size_default=self.gate_size_default, gate_dim1='D1', gate_dim2='D2')
        elif self.node_type == 'spherical':
            node = SphericalModelNode(self.logistic_k, gate, gate_size_default=self.gate_size_default, gate_dim1='D1', gate_dim2='D2')
        elif self.node_type == 'ball':
            node = BallModelNode(self.logistic_k, gate, gate_size_default=self.gate_size_default, gate_dim1='D1', gate_dim2='D2')
        elif self.node_type == 'hyperrectangle':
            node = HyperrectangleModelNode(self.logistic_k, gate, gate_size_default=self.gate_size_default, gate_dim1='D1', gate_dim2='D2')
        else:
            raise ValueError('Node type not recognized. Options are square, and rectangle')
        return node
                        
    def forward(self, x, y=None, detach_logistic_params=False, use_hard_proportions=False, device=0):
        self.set_device(0)
        self.init_output(x) 
        
        for sample_idx in range(len(x)):
            for leaf_idx in range(len(self.nodes)):
                # updates the old regularization that doesnt seem to work well
                # also updates the leaf probabilities
                self.update_output_per_leaf(leaf_idx, sample_idx, x[sample_idx])

        loss = self.output['ref_reg_loss'] + self.output['size_reg_loss'] + self.output['corner_reg_loss'] + self.output['init_reg_loss']

        if use_hard_proportions:
            self.output['leaf_probs'] = torch.tensor(self.get_hard_proportions(x)[:, np.newaxis], dtype=torch.float32).cuda()
        if self.use_log_p:
            self.output['leaf_logp'] = torch.log(self.output['leaf_probs']).clamp(min=-1000) 
        else:
            self.output['leaf_logp'] = (self.output['leaf_probs']) 


        if self.classifier:
            if detach_logistic_params:
                self.output['leaf_logp'] = self.output['leaf_logp'].detach()
            self.output['y_pred'] = torch.sigmoid(self.linear(self.output['leaf_logp'])).squeeze(1)
        
        self.output['feature_diff_reg'] = torch.tensor(0.)
        self.output['emp_reg_loss'] = torch.tensor(0.)
        self.output['log_loss'] = torch.tensor(0.)
        if y is not None:
            self.update_output_feat_diff_and_emp_reg(x, y)
            if self.classifier:
                self.update_output_classification_loss(y)
        loss = loss + self.output['feature_diff_reg'] + self.output['emp_reg_loss'] + self.output['log_loss']
        self.output['loss'] = loss
        return self.output
        
    def init_output(self, x):
        self.output = {'leaf_probs': None,
                  'leaf_logp': None,
                  'y_pred': None,
                  'ref_reg_loss': 0,
                  'size_reg_loss': 0,
                  'init_reg_loss': 0,
                  'emp_reg_loss': 0,
                  'corner_reg_loss': 0,
                  'log_loss': None,
                  'loss': None
                  }
        
        self.output['leaf_probs'] = self.init_leaf_probs(x)

    def init_leaf_probs(self, x): 
        tensor = torch.tensor((), dtype=torch.float32)
        leaf_probs = tensor.new_zeros((len(x), len(self.nodes)))
        if torch.cuda.is_available():
            leaf_probs.cuda(self.device)
        return leaf_probs

    def set_device(self, device_num):
        self.device = device_num

    def update_output_per_leaf(self, leaf_idx, sample_idx, single_sample):
        leaf_node = self.nodes[leaf_idx]
        logp, ref_reg_penalty, init_reg_penalty, size_reg_penalty, corner_reg_penalty = leaf_node(single_sample)
        self.output['ref_reg_loss'] += ref_reg_penalty * self.regularisation_penalty / len(single_sample) 
        self.output['size_reg_loss'] += size_reg_penalty * self.gate_size_penalty / len(single_sample)
        self.output['corner_reg_loss'] += corner_reg_penalty * self.corner_penalty / len(single_sample)
        self.output['init_reg_loss'] += init_reg_penalty * self.init_reg_penalty/ len(single_sample)
        self.output['leaf_probs'][sample_idx, leaf_idx] = logp.exp().sum(dim=0) / single_sample.shape[0]

    def update_output_feat_diff_and_emp_reg(self, x, y):
        pos_mean = 0.
        neg_mean = 0.
        for sample_idx in range(len(y)):
            if y[sample_idx] == 0:
                self.output['emp_reg_loss'] = self.output['emp_reg_loss'] + self.negative_box_penalty * \
                                         torch.abs(self.output['leaf_logp'][sample_idx][0] - np.log(self.neg_proportion_default))/ (len(y) - sum(y))
                neg_mean = neg_mean + self.output['leaf_probs'][sample_idx][0]
            else:
                pos_mean = pos_mean + self.output['leaf_probs'][sample_idx][0]
        # use the average mean to normalize the difference so the square isn't so tiny
        self.output['feature_diff_reg'] = self.feature_diff_penalty * \
                                     -torch.log((((1./(len(y) - sum(y))) * neg_mean - (1./(sum(y))) * pos_mean))**2)

    def update_output_classification_loss(self, y):
        if self.loss_type == "logistic":
            self.output['log_loss'] = self.criterion(self.linear(self.output['leaf_logp']).squeeze(1), y)
        elif self.loss_type == "MSE":
            self.output['log_loss'] = self.criterion(self.output['y_pred'], y)
        else:
            raise NotImplementedError('Only options for classification loss are logistic and MSE') 

    #filter_data from parent doesnt work in this case
    def filter_data(self, data):
        filtered_data = []
        for node in self.nodes:
            filtered_data.append(self.filter_data_at_single_node(data, node))
        return filtered_data

    def get_gates(self):
        gates = []
        for node in self.nodes:
            gates.append(node.get_gate())
        return gates

    def get_gate_tree(self):
        gates = self.get_gates()
        if self.node_type == 'circular':
            tree = gates
        else:
            tree = []
            for gate in gates:
                tree.append(
                    [
                        ['D1', gate[0], gate[1]],
                        ['D2', gate[2], gate[3]]
                    ]
                )
        return tree

    def fix_size_params(self, size):
        for node in self.nodes:
            node.side_length_param.requires_grad = False
       
    def freeze_gate_params(self):
        for node in self.nodes:
            for param in node.parameters():
                param.requires_grad = False

    def get_l1_loss(self, l1_reg_strength):
        return l1_reg_strength * torch.norm(self.linear.weight)

class DisjunctiveDepthOneModel(DepthOneModel):

    def __init__(self, init_gates, model_params):
        super().__init__(init_gates, model_params)
        # disjunction of all gates, so there's only a single feature!
        self.linear = nn.Linear(1, 1)

    def forward(self, x, y=None, detach_logistic_params=False, use_hard_proportions=False, device=0):
        self.set_device(0)
        self.init_output(x) 
        
        self.init_leaf_gates(x)
        for sample_idx in range(len(x)):
            for leaf_idx in range(len(self.nodes)):
                # updates the old regularization that doesnt seem to work well
                # also updates the leaf probabilities
                self.update_output_per_leaf(leaf_idx, sample_idx, x[sample_idx])

        loss = self.output['ref_reg_loss'] + self.output['size_reg_loss'] + self.output['corner_reg_loss'] + self.output['init_reg_loss']

        if use_hard_proportions:
            self.output['leaf_probs'] = torch.tensor(self.get_hard_proportions(x)[:, np.newaxis], dtype=torch.float32).cuda()
        
        # this is the only difference with DepthOneModel forwards
        # we take log of the (smoothed) disjunction to get a single feature
        smooth_disjunction_prop = self.get_smooth_disjunction_proportion(x)
        self.output['leaf_probs'] = smooth_disjunction_prop
        self.output['leaf_logp'] = torch.log(smooth_disjunction_prop).clamp(-1000)


        if self.classifier:
            if detach_logistic_params:
                self.output['leaf_logp'] = self.output['leaf_logp'].detach()
            self.output['y_pred'] = torch.sigmoid(self.linear(self.output['leaf_logp'])).squeeze(1)
        
        self.output['feature_diff_reg'] = torch.tensor(0.)
        self.output['emp_reg_loss'] = torch.tensor(0.)
        self.output['log_loss'] = torch.tensor(0.)
        if y is not None:
            self.update_output_feat_diff_and_emp_reg(x, y)
            if self.classifier:
                self.update_output_classification_loss(y)
        loss = loss + self.output['feature_diff_reg'] + self.output['emp_reg_loss'] + self.output['log_loss']
        self.output['loss'] = loss
        return self.output

    def init_leaf_gates(self, x):
        max_num_cells = np.max(np.array([s.shape[0] for s in x]))
        self.output['leaf_gates'] = torch.zeros([len(x), self.num_gates, max_num_cells])

    def update_output_per_leaf(self, leaf_idx, sample_idx, single_sample):
        leaf_node = self.nodes[leaf_idx]
        logp, ref_reg_penalty, init_reg_penalty, size_reg_penalty, corner_reg_penalty = leaf_node(single_sample)
        self.output['ref_reg_loss'] += ref_reg_penalty * self.regularisation_penalty / len(single_sample) 
        self.output['size_reg_loss'] += size_reg_penalty * self.gate_size_penalty / len(single_sample)
        self.output['corner_reg_loss'] += corner_reg_penalty * self.corner_penalty / len(single_sample)
        self.output['init_reg_loss'] += init_reg_penalty * self.init_reg_penalty/ len(single_sample)
#        self.output['leaf_probs'][sample_idx, leaf_idx] = logp.exp().sum(dim=0) / single_sample.shape[0]
        self.output['leaf_gates'][sample_idx, leaf_idx, :single_sample.shape[0]] = torch.exp(logp)

    def get_smooth_disjunction_proportion(self, x):
        # use .5 * self.num_gates as the center of the sigmoid
        # since this would be the case where a cell is on the boundary
        # for every gate (ie we should have 50 percent chance of being in
        # the disjunction)
        #smooth_disjunction = torch.sigmoid(
        #    (torch.sum(self.output['leaf_probs'], dim=1) - .5 * self.num_gates) \
        #    * self.disjunctive_k 
        #)
        smooth_disjunction = torch.max(self.output['leaf_gates'], dim=1)[0]
#        print(smooth_disjunction)
#        print(smooth_disjunction.shape)
        sample_shapes = torch.tensor([s.shape[0] for s in x], dtype=smooth_disjunction.dtype)
        smooth_disjunction_prop = smooth_disjunction.sum(dim=1)/sample_shapes
#        print('disjunction props:\n', smooth_disjunction_prop)
#        print(smooth_disjunction_prop.shape)
        return smooth_disjunction_prop.reshape(-1, 1)
