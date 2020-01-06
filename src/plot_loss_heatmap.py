import matplotlib.pyplot as plt
import seaborn as sb
import pickle
from utils.TransformParameterParser import TransformParameterParser
from utils.DataInput import DataInput
from utils.DepthOneModel import DepthOneModel
from train_UMAP import *
from utils.GateInitializerClustering import GateInitializer
from plot_loss_near_converged_solution import set_random_seeds
from plot_loss_near_converged_solution import init_data_input


def main(path_to_config, transformer_path):
    params = TransformParameterParser(path_to_config).parse_params()
    set_random_seeds(params)
    data_input = init_data_input(params, transformer_path)
    grid_x, grid_y, size_grid = get_small_grid()
    plotter = HeatMapPlotter(params, data_input, grid_x, grid_y, size_grid)
    fig, _ = plotter.plot_loss_heat_maps_and_data_density() 
    fig.savefig('heatmaps.png')

def get_small_grid():
    return np.arange(0, 11)/10, np.arange(0, 11)/10, np.arange(1, 5)/20

def get_larger_grid():
    return np.arange(1, 20)/20, np.arange(1, 20)/20, np.arange(1, 10)/10

class HeatMapPlotter:
    def __init__(self, params, data_input, grid_x, grid_y, size_grid):
        self.grid_x = grid_x
        self.grid_y = grid_y
        self.size_grid = size_grid
        self.data_input = data_input
        self.params = params

    def plot_loss_heat_maps_and_data_density(self, figsize=5):
        fig, axes = plt.subplots(2, self.size_grid.shape[0], figsize=(self.size_grid.shape[0] * figsize, 2* figsize))
        for s, size in enumerate(self.size_grid):
            self.plot_single_heat_map(axes[0][s], size)
            axes[0][s].set_title('size=%.2f' %size)
        self.plot_data_density(axes[int(axes.shape[0]/2)][1])
        return fig, axes

    def plot_single_heat_map(self, axis, size):
        loss_grid = np.zeros([self.grid_x.shape[0], self.grid_y.shape[0]])
        for i, x_pos in enumerate(self.grid_x):
            for j, y_pos in enumerate(self.grid_y):
                model = self.get_model_with_set_position_and_size(x_pos, y_pos, size)
                output = model(self.data_input.x_tr, self.data_input.y_tr)
                loss_grid[j, i] = output['log_loss']
        ax = sb.heatmap(loss_grid, ax=axis, xticklabels=self.grid_x, yticklabels=self.grid_y, fmt='f', vmin=.2, vmax=.7)
        ax.invert_yaxis()


    def plot_data_density(self, axis):
        sb.kdeplot(self.get_data_for_plot(), ax=axis)
    
    def get_data_for_plot(self, subsample=1000):
        numpy_data = [x_tr.cpu().detach().numpy() for x_tr in self.data_input.x_tr]
        subsampled_data = [data[0:subsample] for data in numpy_data]
        return np.concatenate(subsampled_data)


    def get_model_with_set_position_and_size(self, x_pos, y_pos, size):
        model = DepthOneModel(self.get_init_gate(x_pos, y_pos, size), self.params['model_params'])
        train_params = self.params['train_params']
        fit_classifier_params(model, self.data_input,\
            train_params['learning_rate_classifier'],
            l1_reg_strength=train_params['l1_reg_strength'])
        return model

    def get_init_gate(self, x_pos, y_pos, size):
        return [[
            ['D1', x_pos - size/2, x_pos + size/2],
            ['D2', y_pos - size/2, y_pos + size/2]
        ]]

    
if __name__ == '__main__':
    path_to_configs = '../configs/umap_default.yaml'
    transformer_path = '../output/repeated_init_testing_grid/transformer.pkl'
    main(path_to_configs, transformer_path)
