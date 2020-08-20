import umap 
from sklearn.decomposition import PCA
from openTSNE import TSNE

#TODO move num_components into transform parameters as well as subsample params
class DataTransformerFactory:
    def __init__(self, params, random_seed):
        self.params = params
        self.transform_type = self.params['transform_type']
        self.random_seed = random_seed

    def manufacture_transformer(self):
        if self.transform_type == 'umap':
            transformer = self.manufacture_umapper()
        elif self.transform_type == 'pca':
            transformer = self.manufacture_pcaer()
        elif self.transform_type == 'tsne':
            transformer = self.manufacture_tsneer()
        elif self.transform_type == 'identity':
            transformer = self.manufacture_identityer()
        else:
            raise ValueError('Transformer type %s not recognized' %self.transform_type)
        return transformer

    def manufacture_umapper(self):
        umapper =\
            umap.UMAP(
                n_neighbors = self.params['umap_params']['n_neighbors'],
                min_dist = self.params['umap_params']['min_dist'],
                n_components = self.params['embed_dim'],
                random_state = self.random_seed
            )
        return umapper

    def manufacture_pcaer(self):
        pcaer = PCA(n_components = self.params['embed_dim'])
        return pcaer

    def manufacture_tsneer(self):
        tsneer = TSNEWrapper(
            self.params,
            self.random_seed
        )
        return tsneer

    def manufacture_identityer(self):
        return IdentityTransformer()

class IdentityTransformer:
    def __init__(self):
        return
    
    def fit(self, data):
        return data

    def transform(self, data):
        return data

class TSNEWrapper:
    
    def __init__(self, params, random_seed):
        self.tsneer = TSNE(
            n_components=params['embed_dim'],
            random_state = random_seed
        )

    def fit(self, data):
        self.embedding = self.tsneer.fit(data)
    
    def transform(self, data):
        new_embedded_data = self.embedding.transform(data)
        return new_embedded_data
