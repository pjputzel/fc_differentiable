import umap
import numpy as np
import time
import os
import pickle

def embed_new_sample(umapper, new_sample):
    return umapper.transform(new_sample)

def embed_samples_with_labels(umapper, data, labels, num_cells=1e5):
    data = [sample[0:int(num_cells)] for sample in data]
    embeddings_with_labels = [np.concatenate([umapper.transform(sample), labels[i] * np.ones([sample.shape[0], 1])], axis=1) for i, sample in enumerate(data)]
    return embeddings_with_labels
    

'''
saves embeddings using a single trained umapper located in a pkl file at umapper path
saves the labels to each cell for convenience of plotting, does NOT use labels to construct
the embedding.
'''
def make_and_save_per_sample_embeddings_with_labels(umapper_path, data_path, labels_path, savepath):
    with open(umapper_path, 'rb') as f:
        umapper = pickle.load(f)    
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    with open(labels_path, 'rb') as f:
        labels = pickle.load(f)

    embeddings_with_labels = embed_samples_with_labels(umapper, data, labels)
    with open(savepath, 'wb') as f:
        pickle.dump(embeddings_with_labels, f)
    return embeddings_with_labels
    


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

def make_and_save_UMAP_embeddings_in_grid(savedir, data_path, n_neighbors_grid, min_dist_grid, subsample_cells_to=1e5, random_state=1, pool_data=True):
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
        if pool_data:
            catted_data = np.concatenate(data)
            np.random.shuffle(catted_data)
        else:
            catted_data = data
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
    ### For using a pretrained umapper to embed a list of samples
    umapper_path = '../output/UMAP_embeddings/num_neighbors=15_min_dist=0.10.pkl_umapper.pkl'
    data_path = '../data/cll/x_UMAP_dev_FIXED.pkl'
    labels_path = '../data/cll/y_UMAP_dev.pkl'
    savepath = '../output/UMAP_embeddings/Per_Sample_Embeddings_num_neighbors=15_min_dist=0.10.pkl'
    make_and_save_per_sample_embeddings_with_labels(umapper_path, data_path, labels_path, savepath)


    ### For simple GMM data (not the synthethic data from the paper, a new dataset I made to play with UMAP)
    #savedir = '../output/UMAP_embeddings/Simple_GMM_Embeddings'
    #data_path = '../data/synth/diag_var_gmm_data.pkl'
    #n_neighbors_grid = [5, 15, 50, 100, 200, 300]
    #min_dist_grid = [0., .1, .2, .3, .4, .5]
    #make_and_save_UMAP_embeddings_in_grid(savedir, data_path, n_neighbors_grid, min_dist_grid, pool_data=False)

    ### For Dev data embeddings
    #savedir = '../output/UMAP_embeddings'
    #data_path = '../data/cll/x_UMAP_dev_FIXED.pkl'
    #n_neighbors_grid = [5, 15, 50, 100, 200, 300]
    #min_dist_grid = [0., .1, .2, .3, .4, .5]
    #make_and_save_UMAP_embeddings_in_grid(savedir, data_path, n_neighbors_grid, min_dist_grid)




