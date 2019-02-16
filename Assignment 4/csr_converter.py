from scipy.sparse import csr_matrix
import numpy as np

'''
This function get the candidate list and return a list of their feature vectors
'''
def get_feature_vectors_list(candidates_list):
    features_list = []
    for candidate in candidates_list:
        features_list.append(candidate.features)
    return features_list

'''
This function get the candidates list and the feature map and return the sparse representation of the
candidates's feature vectors
'''
def create_sparse_feature_vectors(candidates_list, features_map):
    rows = []
    cols = []
    feature_counter = 0
    index = 0
    feature_vectors_list = get_feature_vectors_list(candidates_list)
    for feature_vec in feature_vectors_list:
        for feature in feature_vec:
            rows.append(index)
            cols.append(feature)
            feature_counter += 1
        index += 1
    data = np.ones((feature_counter),dtype=int)
    sparse_matrix = csr_matrix(arg1=(data,(rows,cols)),shape=(index,len(features_map)))

    return sparse_matrix

