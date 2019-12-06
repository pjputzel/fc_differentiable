from utils.HeuristicInitializer import HeuristicInitializer
from utils.GridHeuristicInitializer import GridHeuristicInitializer

class MultipleGateInitializerHeuristic:
    def __init__(self, data_input, node_type, params):
        self.pos_data = data_input.get_pos_cat_tr_data()
        self.neg_data = data_input.get_neg_cat_tr_data()
        self.node_type = node_type
        for key in params:
            setattr(self, key, params[key])

        self.gates = []

        # holds same info as self.gates, but in
        # the format used by the model node code
        self.gate_tree = []

    def init_next_gate(self):
        cur_acceptable_gates_func = self.get_cur_acceptable_gates_func()
        self.init_new_initializer(cur_acceptable_gates_func)
        next_gate = self.cur_gate_initializer.get_heuristic_gates()[0]
        self.gates.append(next_gate)
        if next_gate is None:
            return None
        next_gate_tree = self.convert_gate_to_tree_format(next_gate)
        self.gate_tree.append(next_gate_tree)
        print('NEXT GATE IN INITALIZER IS:', next_gate)
        return next_gate_tree

    def get_cur_acceptable_gates_func(self):
        gates = self.gates
        def cur_acceptable_gates_func(trial_gate):
            # not actually a gate
            if (trial_gate[0] == trial_gate[1]) or (trial_gate[2] == trial_gate[3]):
                return False
            # may need to modify and add more control
            # flow here for square gates 
            for gate in gates:
                if MultipleGateInitializerHeuristic.overlaps(gate, trial_gate):
                    return False
            return True
        return cur_acceptable_gates_func

    @staticmethod
    def overlaps(gate, trial_gate):
        if ((trial_gate[0] >= gate[0]) and (trial_gate[0] < gate[1])):
            if MultipleGateInitializerHeuristic.overlaps_dim2(gate, trial_gate):
                return True
        if ((trial_gate[1] > gate[0]) and (trial_gate[1] <= gate[1])):
            if MultipleGateInitializerHeuristic.overlaps_dim2(gate, trial_gate):
                return True

        if ((gate[0] >= trial_gate[0]) and (gate[0] < trial_gate[1])):
            if MultipleGateInitializerHeuristic.overlaps_dim2(trial_gate, gate):
                return True
        if ((gate[1] > trial_gate[0]) and (gate[1] <= trial_gate[1])):
            if MultipleGateInitializerHeuristic.overlaps_dim2(trial_gate, gate):
                return True
        
        return False    
    @staticmethod
    def overlaps_dim2(gate, trial_gate):
        if ((trial_gate[2] >= gate[2]) and (trial_gate[2] < gate[3])):
            return True
        if ((trial_gate[3] > gate[2]) and (trial_gate[3] <= gate[3])):
            return True
        return False
    
    def init_new_initializer(self, cur_acceptable_gates_func): 
        if self.init_type == 'heuristic_corner':
            gate_initializer = HeuristicInitializer(\
                self.node_type,
                [[0, 1]],
                self.pos_data,
                self.neg_data,
                num_gridcells_per_axis=self.num_gridcells_per_axis,
                is_gate_acceptable_func=cur_acceptable_gates_func
            )
        elif self.init_type == 'heuristic_grid':
            gate_initializer = GridHeuristicInitializer(\
                self.node_type,
                [[0,1]],
                self.pos_data,
                self.neg_data,
                num_gridcells_per_axis=self.num_gridcells_per_axis,
                is_gate_acceptable_func=cur_acceptable_gates_func
        )
        self.cur_gate_initializer = gate_initializer
        return gate_initializer
    
    def convert_gate_to_tree_format(self, gate):
        return [[u'D1', gate[0], gate[1]], [u'D2', gate[2], gate[3]]]
