import matplotlib.pyplot as plt
import numpy as np
import pickle
import os
import numpy as np

'''
plots saved embeddings for a range of different parameter values
'''                
def plot_umap_embeddings(umap_embeddings_dir, n_neighbors_grid, min_dist_grid, num_cells_per_sample=1e5, num_dims=2, marker_size=.001):
    data_for_plot = np.zeros([len(n_neighbors_grid), len(min_dist_grid), int(num_cells_per_sample), num_dims]) 
    for row, n_neighbors in enumerate(n_neighbors_grid):
        for col, min_dist in enumerate(min_dist_grid):
            filename = get_umap_embedding_filename(n_neighbors, min_dist)
            filepath = os.path.join(umap_embeddings_dir, filename)    
            with open(filepath, 'rb') as f:
                cur_embedding = pickle.load(f)
            data_for_plot[row, col] = cur_embedding
    fig, axes = plot_data_array(data_for_plot, labels=[n_neighbors_grid, min_dist_grid], marker_size=marker_size)
    fig.savefig(os.path.join(umap_embeddings_dir, 'umap_embeddings.png'))

def get_umap_embedding_filename(n_neighbors, min_dist):
    return 'num_neighbors=%d_min_dist=%.2f.pkl' %(n_neighbors, min_dist)

'''
plots a 2d np array of 2d np data arrays on a grid of axes
The position in the first two dimensions of the np array determine
the position in the grid of axes.
'''
def plot_data_array(data, colors=None, labels=None, fig_unit_len=5, marker_size=.001):
    n_rows = data.shape[0]
    n_cols = data.shape[1]
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_unit_len * n_cols, fig_unit_len * n_rows))
    for row in range(n_rows):
        for col in range(n_cols):
            cur_axis = axes[row, col]
            cur_data = data[row, col]
            color = colors[row, col] if colors else 'b'
            cur_axis.scatter(cur_data[:, 0], cur_data[:, 1], color=color, s=marker_size)
    
    if labels:
        add_labels_to_axes_grid(labels, axes)    

    return fig, axes
 
def add_labels_to_axes_grid(labels, axes):
    row_labels = labels[0]
    col_labels = labels[1]
    for row, label in enumerate(row_labels):
        axes[row, 0].set_ylabel('n_neighbors=' + str(label))
    for col, label in enumerate(col_labels):
        axes[0, col].set_title('min_dist=' + str(label))


if __name__ == '__main__':
    embeddings_dir = '../../output/UMAP_embeddings/Simple_GMM_Embeddings/'
    n_neighbors_grid = [5, 15, 50, 100, 200, 300]
    min_dist_grid = [0.00, 0.10, 0.20, 0.30, 0.40, 0.50]
    plot_umap_embeddings(embeddings_dir, n_neighbors_grid, min_dist_grid, num_cells_per_sample=1000, marker_size=.1)
