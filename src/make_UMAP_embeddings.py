import umap
import numpy as np
import time
import os
import pickle

def make_UMAP_embedding(catted_data, n_neighbors, min_dist, random_state):
    umapper = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=random_state)
    embedding = umapper.fit_transform(catted_data)
    return umapper, embedding

def save_embedding(embedding, savepath):
    with open(savepath, 'wb') as f:
        pickle.dump(embedding, f)

def save_umapper(umapper, savepath):
    with open(savepath, 'wb') as f:
        pickle.dump(umapper, f)

def make_and_save_UMAP_embeddings_in_grid(savedir, data_path, n_neighbors_grid, min_dist_grid, subsample_cells_to=1e5, random_state=1):
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
        catted_data = np.concatenate(data)
        np.random.shuffle(catted_data)
        catted_data = catted_data[0:int(subsample_cells_to)]
    for n_neighbors in n_neighbors_grid:
        for min_dist in min_dist_grid:
            start_time = time.time()
            print('n_neighbors: %d, min_dist %.2f' %(n_neighbors, min_dist))
            umapper, embedding = make_UMAP_embedding(catted_data, n_neighbors, min_dist, random_state)
            savepath = os.path.join(savedir, 'num_neighbors=%d_min_dist=%.2f.pkl' %(n_neighbors, min_dist))
            save_embedding(embedding, savepath)
            savepath = savepath + '_umapper.pkl'
            save_umapper(umapper, savepath)
            print('time taken: %d' %(time.time() - start_time))

if __name__ == '__main__':
    savedir = '../output/UMAP_embeddings'
    data_path = '../data/cll/x_UMAP_dev_FIXED.pkl'
    n_neighbors_grid = [5, 15, 50, 100, 200, 300]
    min_dist_grid = [0., .1, .2, .3, .4, .5]
    make_and_save_UMAP_embeddings_in_grid(savedir, data_path, n_neighbors_grid, min_dist_grid)




