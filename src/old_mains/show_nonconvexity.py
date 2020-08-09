import umap
import warnings
import pickle
import numpy as np
import matplotlib.pyplot as plt
from utils.TransformParameterParser import TransformParameterParser
from utils.DataInput import DataInput
from utils.MultipleGateInitializerHeuristic import MultipleGateInitializerHeuristic
from utils.DepthOneModel import DepthOneModel
from utils.DataAndGatesPlotter import DataAndGatesPlotterDepthOne
from utils.DataTransformerFactory import DataTransformerFactory
from train_UMAP import run_train_model
import torch
import os
import time
import numpy as np
from collections import namedtuple

# Runs convexity test for simple 2-d gate problem


def run_convexity_test():
    # just copy loading code from loss plots, although maybe make it cleaner
    data_input = get_basic_data_input()
    parameter_setting_grid = get_basic_parameter_dictionary_grid()
    test_convexity(parameter_setting_grid, data_input)

def get_basic_data_input():
    scale1 = 3
    scale2 = 1
    num_points = 200
    x1 = np.random.normal(loc=.5, scale=scale1, size=(int(num_points/2), 1))
    x2 = np.random.normal(loc=1.5, scale=scale2, size=(int(num_points/2), 1))
    data_input = namedtuple('x_tr', 'y_tr')
    x_all = np.concatenate([x1, x2])
    idxs = np.random.permutation(np.arange(num_points))
    x_all = x_all[idxs]
    y_all = np.concatenate([np.zeros([int(num_points/2), 1]), np.ones([int(num_points/2), 1])])[idxs]
    data_input.x_tr = x_all
    data_input.y_tr = y_all
    return data_input        
    
    

def get_basic_parameter_dictionary_grid():
    a_grid = np.linspace(-10, 10, 10)
    b_grid = np.linspace(-10, 10, 10)
    theta_0_grid = np.linspace(-10, 10, 10)
    theta_1_grid = np.linspace(-10, 10, 10)
    dict_grid = []
    for i in range(10):
        cur_dict = {\
            'a': a_grid[i],
            'b': b_grid[i],
            'theta0': theta_0_grid[i],
            'theta1': theta_1_grid[i]
        }
        dict_grid.append(cur_dict)
    return dict_grid
        


# use a config with high subsampling
def test_convexity(parameter_setting_grid, data_input): 
    is_convex = True
    for parameter_setting in parameter_setting_grid:
        cur_hessian = get_hessian(parameter_setting, data_input)
        print(cur_hessian)
        eigenvalues, _ = np.linalg.eig(cur_hessian)
        for eigenvalue in eigenvalues:
            if eigenvalue <= 0:
                print('Nonconvex at setting: ', parameter_setting)
                is_convex = False
    return is_convex

def get_hessian(parameter_setting, data_input):
    hessian = np.zeros([len(parameter_setting), len(parameter_setting)])
    for i, param1 in enumerate(parameter_setting.keys()):
        for j, param2 in enumerate(parameter_setting.keys()):
            hessian[i, j] = compute_dLoss_dparam1_dparam2(data_input, parameter_setting, param1, param2)
    return hessian

def compute_dLoss_dparam1_dparam2(data_input, parameter_setting, param1, param2):
    if param1 == 'a':
        return compute_dLoss_da_dparam2(data_input, parameter_setting, param2)
    elif param1 == 'b':
        return compute_dLoss_db_dparam2(data_input, parameter_setting, param2)
    elif param1 == 'theta0':
        return compute_dLoss_dtheta0_dparam2(data_input, parameter_setting, param2)
    elif param1 == 'theta1':
        return compute_dLoss_dtheta1_dparam2(data_input, parameter_setting, param2)

def compute_dLoss_da_dparam2(data_input, parameter_setting, param2):
    if param2 == 'a':
        return compute_dLoss_da_da(data_input, parameter_setting)
    if param2 == 'b':
        return compute_dLoss_da_db(data_input, parameter_setting)
    if param2 == 'theta0':
        return compute_dLoss_da_dtheta0(data_input, parameter_setting)
    if param2 == 'theta1':
        return compute_dLoss_da_dtheta1(data_input, parameter_setting)

def compute_dLoss_db_dparam2(data_input, parameter_setting, param2):
    if param2 == 'a':
        return compute_dLoss_da_db(data_input, parameter_setting)
    if param2 == 'b':
        return compute_dLoss_db_db(data_input, parameter_setting)
    if param2 == 'theta0':
        return compute_dLoss_db_dtheta0(data_input, parameter_setting)
    if param2 == 'theta1':
        return compute_dLoss_db_dtheta1(data_input, parameter_setting)

def compute_dLoss_dtheta0_dparam2(data_input, parameter_setting, param2):
    if param2 == 'a':
        return compute_dLoss_da_dtheta0(data_input, parameter_setting)
    if param2 == 'b':
        return compute_dLoss_db_dtheta0(data_input, parameter_setting)
    if param2 == 'theta0':
        return compute_dLoss_dtheta0_dtheta0(data_input, parameter_setting)
    if param2 == 'theta1':
        return compute_dLoss_dtheta0_dtheta1(data_input, parameter_setting)
    
def compute_dLoss_dtheta1_dparam2(data_input, parameter_setting, param2):
    if param2 == 'a':
        return compute_dLoss_da_dtheta1(data_input, parameter_setting)
    if param2 == 'b':
        return compute_dLoss_db_dtheta1(data_input, parameter_setting)
    if param2 == 'theta0':
        return compute_dLoss_dtheta0_dtheta1(data_input, parameter_setting)
    if param2 == 'theta1':
        return compute_dLoss_dtheta1_dtheta1(data_input, parameter_setting)

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def compute_feature(difference_s, difference_f, s=1e3):
    return sigmoid(difference_s * s) * (1 - sigmoid(difference_f * s))

def dsigmoid_dx(x):
    return sigmoid(x) * (1 - sigmoid(x))

def dsigmoid_dx_dx(x):
    return dsigmoid_dx(x) * (1 - 2 * sigmoid(x))

def compute_features_and_theta_differences(data_input, parameter_setting): 
    differences_s = data_input.x_tr - parameter_setting['theta0']
    differences_f = data_input.x_tr - parameter_setting['theta1']
    features = compute_feature(differences_s, differences_f)
    return differences_s, differences_f, features

def compute_dfeature_dtheta0(data_input, parameter_setting, s=1e3):
    differences_s, differences_f, features = compute_features_and_theta_differences(data_input, parameter_setting)
    z = dsigmoid_dx(s * (differences_s))
    return -s * z * (1 - sigmoid(s * differences_f))

def compute_dfeature_dtheta1(data_input, parameter_setting, s=1e3):
        
    differences_s, differences_f, features = compute_features_and_theta_differences(data_input, parameter_setting)
    z = dsigmoid_dx(s * (differences_f))
    return s * z * (sigmoid(s * differences_s))

def compute_dfeature_dtheta0_dtheta1(data_input, parameter_setting, s=1e3):
    differences_s, differences_f, features = compute_features_and_theta_differences(data_input, parameter_setting)
    z_f = dsigmoid_dx(s * (differences_f))
    z_s = dsigmoid_dx(s * differences_s)
    return -(s**2) * z_f * z_s
    

def compute_dLoss_da_da(data_input, parameter_setting):
    differences_s, differences_f, features = compute_features_and_theta_differences(data_input, parameter_setting)
    dsigmoid_dx_dxs = dsigmoid_dx_dx(parameter_setting['a'] * features + parameter_setting['b'])
    return np.sum((2 * data_input.y_tr - 1) * features**2 * dsigmoid_dx_dxs)

def compute_dLoss_da_db(data_input, parameter_setting):
    differences_s, differences_f, features = compute_features_and_theta_differences(data_input, parameter_setting)
    dsigmoid_dx_dxs = dsigmoid_dx_dx(parameter_setting['a'] * features + parameter_setting['b'])
    return np.sum((2 * data_input.y_tr - 1) * features * dsigmoid_dx_dxs)
    
def compute_dLoss_da_dtheta0(data_input, parameter_setting):
    differences_s, differences_f, features = compute_features_and_theta_differences(data_input, parameter_setting)
    term1_i = (2 * data_input.y_tr - 1) * (1 + parameter_setting['a'] * features * (1 - 2 * sigmoid(parameter_setting['a'] * features + parameter_setting['b'])))
    term2_i = dsigmoid_dx(parameter_setting['a'] * features + parameter_setting['b']) * compute_dfeature_dtheta0(data_input, parameter_setting, s=1e3)
    return np.sum(term1_i * term2_i)

def compute_dLoss_da_dtheta1(data_input, parameter_setting):
    differences_s, differences_f, features = compute_features_and_theta_differences(data_input, parameter_setting)
    term1_i = (2 * data_input.y_tr - 1) * (1 + parameter_setting['a'] * features * (1 - 2 * sigmoid(parameter_setting['a'] * features + parameter_setting['b'])))
    term2_i = dsigmoid_dx(parameter_setting['a'] * features + parameter_setting['b']) * compute_dfeature_dtheta1(data_input, parameter_setting, s=1e3)
    return np.sum(term1_i * term2_i)

def compute_dLoss_db_db(data_input, parameter_setting):
    differences_s, differences_f, features = compute_features_and_theta_differences(data_input, parameter_setting)
    dsigmoid_dx_dxs = dsigmoid_dx_dx(parameter_setting['a'] * features + parameter_setting['b'])
    return np.sum((2 * data_input.y_tr - 1) * dsigmoid_dx_dxs)

def compute_dLoss_db_dtheta0(data_input, parameter_setting):
    differences_s, differences_f, features = compute_features_and_theta_differences(data_input, parameter_setting)
    dsigmoid_dx_dxs = dsigmoid_dx_dx(parameter_setting['a'] * features + parameter_setting['b'])
    dfeats_dtheta0 = compute_dfeature_dtheta0(data_input, parameter_setting, s=1e3)
    return np.sum(parameter_setting['a'] * (2 * data_input.y_tr - 1) * dfeats_dtheta0 * dsigmoid_dx_dxs)

def compute_dLoss_db_dtheta1(data_input, parameter_setting):
    differences_s, differences_f, features = compute_features_and_theta_differences(data_input, parameter_setting)
    dsigmoid_dx_dxs = dsigmoid_dx_dx(parameter_setting['a'] * features + parameter_setting['b'])
    dfeats_dtheta1 = compute_dfeature_dtheta1(data_input, parameter_setting, s=1e3)
    return np.sum(parameter_setting['a'] * (2 * data_input.y_tr - 1) * dfeats_dtheta1 * dsigmoid_dx_dxs)

def compute_dLoss_dtheta0_dtheta0(data_input, parameter_setting, s=1e3):
    differences_s, differences_f, features = compute_features_and_theta_differences(data_input, parameter_setting)
    dsigmoid_dx_dxs_feat = dsigmoid_dx_dx(parameter_setting['a'] * features + parameter_setting['b'])
    dsigmoid_dx_dxs_cut = dsigmoid_dx_dx(s * (differences_s))
    df_dtheta0_dtheta0 = s**2 * dsigmoid_dx_dxs_cut * ( 1 - sigmoid(s * differences_f))
    dfeats_dtheta1 = compute_dfeature_dtheta1(data_input, parameter_setting, s=1e3)
    dfeats_dtheta0 = compute_dfeature_dtheta0(data_input, parameter_setting, s=1e3)
    term1 = parameter_setting['a'] * dsigmoid_dx_dxs_feat * dfeats_dtheta0 ** 2
    term2 = dsigmoid_dx(parameter_setting['a'] * features + parameter_setting['b']) * df_dtheta0_dtheta0
    return np.sum(parameter_setting['a'] * (2 * data_input.y_tr - 1) * (term1 + term2))

def compute_dLoss_dtheta0_dtheta1(data_input, parameter_setting, s=1e3):
    differences_s, differences_f, features = compute_features_and_theta_differences(data_input, parameter_setting)
    dsigmoid_dx_dxs_feat = dsigmoid_dx_dx(parameter_setting['a'] * features + parameter_setting['b'])
    dsigmoid_dx_dxs_cut = dsigmoid_dx_dx(s * (differences_s))
    df_dtheta0_dtheta0 = s**2 * dsigmoid_dx_dxs_cut * ( 1 - sigmoid(s * differences_f))
    dfeats_dtheta1 = compute_dfeature_dtheta1(data_input, parameter_setting, s=1e3)
    dfeats_dtheta0 = compute_dfeature_dtheta0(data_input, parameter_setting, s=1e3)
    z = dsigmoid_dx(parameter_setting['a'] * features + parameter_setting['b'])
    dfeats_dtheta0_dtheta1 = compute_dfeature_dtheta0_dtheta1(data_input, parameter_setting)
    term2 = (1 + (1 - 2 * sigmoid(parameter_setting['a'] * features + parameter_setting['b'])) * parameter_setting['a'] * features)
    return np.sum(parameter_setting['a'] * (2 * data_input.y_tr - 1) * z * dfeats_dtheta0_dtheta1 * (term2))

def compute_dLoss_dtheta1_dtheta1(data_input, parameter_setting, s=1e3):
    differences_s, differences_f, features = compute_features_and_theta_differences(data_input, parameter_setting)
    dfeats_dtheta1 = compute_dfeature_dtheta1(data_input, parameter_setting, s=1e3)
    df_dtheta1_dtheta1 = -s * (1 - 2 * sigmoid(s * (differences_f))) * dfeats_dtheta1
    dsigmoid_dx_dxs_feat = dsigmoid_dx_dx(parameter_setting['a'] * features + parameter_setting['b'])
    z = dsigmoid_dx(parameter_setting['a'] * features + parameter_setting['b'])
    term1 = parameter_setting['a'] * dsigmoid_dx_dxs_feat * dfeats_dtheta1 ** 2
    term2 = dsigmoid_dx(parameter_setting['a'] * features + parameter_setting['b']) * df_dtheta1_dtheta1
    return np.sum(parameter_setting['a'] * (2 * data_input.y_tr - 1) * (term1 + term2))
    
    
if __name__ == '__main__':
    run_convexity_test() 
    
    
