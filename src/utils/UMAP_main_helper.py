import sklearn.model_selection import train_test_split 
from utils.utils_load_data import normalize_x_list

#TODO: After writing the DataInput class refactor to use that in these functions



def load_and_split_data(data_handling_params):
    with open(data_handling_params['data_path'], 'rb') as f:
        data = pickle.load(f)
    with open(data_handling_params['labels_path'], 'rb') as f:
        labels = pickle.load(f)
    return train_test_split(data, labels, test_size=data_handling_params['test_percent'])

'''
TODO: implement cell subsampling if needed
'''
def embed_data(umapper, tr_data, te_data, embed_dim, cells_to_subsample=1e5):
    umapper.fit(np.concatenate(tr_data), embed_dim=embed_dim)
    tr_embed = [umapper.transform(tr_sample) for tr_sample in tr_data]
    te_embed = [umapper.transform(te_sample) for te_sample in te_data]
    return tr_embed, te_embed

def normalize_data(tr_data, te_data):
    tr_data_normalized, offset, scale = normalize_x_list(tr_data)
    te_data_normalized = normalize_x_list(te_data, offset=offset, scale=scale)
    return tr_data_normalized, te_data_normalized
    
def initialize_gates(cluster_memberships_tr, gate_init_params):
    init_gates = []
    clusters = np.unique(cluster_memberships_tr)
    for cluster in clusters:
        # can add percentile low/high here if needed unpacked from gate init params
        init_gates.append(get_initial_gate_per_cluster(cluster))
    return init_gates
        
'''
TODO: update for arbitrary dimensions
'''
def get_initial_gate_per_cluster(cluster, percentile_low=.05, percentile_high=.95):
    sorted_x = np.sort(cluster[:, 0])
    sorted_y = np.sort(cluster[:, 1])
    low_idx = int(percentile_high * cluster.shape[0])
    high_idx = int(percentile_low * cluster.shape[0])
    return [sorted_x[low_idx], sorted_x[high_idx], 
            sorted_y[low_idx], sorted_y[high_idx]]
    

def run_train_multi_gate_model(init_gates, train_params, tr_data):
    init_gates_tree = convert_init_gates_to_tree(init_gates)
    # model initialization requires a ref tree, but we don't have one for the model with UMAP. Since we aren't using the ref tree for anything we can just pass a fake one.
    dummy_ref_tree = init_gates_tree
    model = 

    


def eval_model(model, te_data):
    pass

def save_results(results):
    pass


 

