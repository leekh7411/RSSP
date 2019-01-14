import os
import h5py
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import scipy.sparse as sp
from global_dict import DATASET_PATH, HYPER_PARAMS
from preprocess import *

def load_file_list(dirname, ftype):
    file_list = []    
    filenames = os.listdir(dirname)
    file_extensions = set(['.{}'.format(ftype)])
    for filename in filenames:
        ext = os.path.splitext(filename)[-1]
        if ext in file_extensions:
            full_filename = os.path.join(dirname, filename)
            file_list.append(full_filename)
        
    return file_list


def onehot_RNA_dict():
    dict = defaultdict(lambda: np.array([.25,.25,.25,.25]))
    #dict = defaultdict(lambda: np.array([-1,-1,-1,-1]))
    dict["A"] = np.array([1,0,0,0])
    dict["a"] = np.array([1,0,0,0])
    dict["C"] = np.array([0,1,0,0])
    dict["c"] = np.array([0,1,0,0])
    dict["G"] = np.array([0,0,1,0])
    dict["g"] = np.array([0,0,1,0])
    dict["U"] = np.array([0,0,0,1])
    dict["u"] = np.array([0,0,0,1])
    return dict


def get_graph_matrix(fname, max_length):
    f = open(fname,"r")
    f_lines = f.readlines()
    graph_structure = []
    onehot_RNA = onehot_RNA_dict()
    
    sequences = []
    adjacency_rows = []
    adjacency_cols = []
        
    # Load and split data in .bpseq file type
    for line in f_lines:
        if line[0] not in set(["#"," "]):
            idx, seq, link_idx = line.split()
            idx = int(idx)
            link_idx = int(link_idx)
            
            # save sequence letter
            sequences.append(seq)
            
            # save adjacency matrix coordinates
            if link_idx != 0:
                adjacency_rows.append(idx)
                adjacency_cols.append(link_idx)
                      
        
    
    # make data list (connection means 1)
    adjacency_data = np.ones(len(adjacency_rows))
    
    
    # convert to sparse_matrix format for save memory storage
    rows = np.array(adjacency_rows)
    cols = np.array(adjacency_cols)
    N    = max_length
    adjacency_matrix = sp.coo_matrix((adjacency_data, (rows,cols)), shape=(N,N))   
    
    
    # preprocess sequence data
    feature_matrix = []
    sequence_length = len(sequences)
    for i in range(N):
        if i < sequence_length:
            seq = sequences[i]
        else:
            seq = 'N'
        
        onehot_seq = onehot_RNA[seq]
        feature_matrix.append(onehot_seq)
    
    # convert to sparse_matrix format for save memory storage
    feature_matrix = np.array(feature_matrix)
    feature_matrix = sp.coo_matrix(feature_matrix)
    
    return feature_matrix, adjacency_matrix, sequence_length

def matrix_normalization(adj):
    adj = sp.coo_matrix(adj)
    adj_ = adj 
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    return adj_normalized

def backbone_matrix(N, connection_count):
    row = []
    col = []
    data = []
    cont_range = connection_count-1
    for i in range(N-1):
        row.append(i)
        col.append(i+1)
        if i < (cont_range-1):
            data.append(1)
        else:
            data.append(.25)
            
    data = np.array(data)
    row  = np.array(row)
    col  = np.array(col)
    
    shift_matrix = sp.coo_matrix((data, (row,col)), shape=(N,N))
    ld_shift = shift_matrix
    ru_shift = shift_matrix.T
    connect_matrix = ld_shift + ru_shift
    
    return connect_matrix


def preprocess_and_save(TEXT_LOGS=False, PLT_LOGS=False):
    
    fpath      = DATASET_PATH["PDB"]
    file_list  = load_file_list(fpath, ftype="bpseq")
    max_length = HYPER_PARAMS["max_seq_length"]
    npz_path   = DATASET_PATH["PDB_npz"]
    
    # print logs
    print("\nPREPROCESS and SAVE")
    print("- dataset path :", fpath)
    print("- loaded file list length :", len(file_list))
    print("- default sequence maximum length :", max_length)
    
    print("- preprocess as graph matrix ...")
    x_mat_list = []
    a_mat_list = []
    s_len_list = []
    y_mat_list = []
    
    for i, fname in enumerate(file_list):
        feature_mat, adj_mat, seq_length = get_graph_matrix(fname, max_length)
        if TEXT_LOGS and i % 100 == 0:
            print("\tFile[{}/{}] >> path : {} & sequence length : {}".format(i+1,len(file_list),fname, seq_length))
                    
        # add default link(sequential link)
        backbone_mat = backbone_matrix(max_length, seq_length)
        adj_mat += backbone_mat
        
        # normalization (used to input data)
        A = matrix_normalization(backbone_mat) # without secondary structure link for predict structure as an adjacency matrix
        X = feature_mat # sequence feature for predict secondary structure
        Y = (adj_mat > 0) # original secondary structure adjacency matrix 
        
        x_mat_list.append(X)
        a_mat_list.append(A)
        y_mat_list.append(Y)
        s_len_list.append(seq_length)
    
    
    np.savez(npz_path,
             X = np.array(x_mat_list), 
             A = np.array(a_mat_list),
             Y = np.array(y_mat_list), 
             L = np.array(s_len_list))
    
    print('- preprocessed dataset saved at {}'.format(npz_path))
    
    if PLT_LOGS:
        plt.figure(figsize=(10,5))
        plt.subplot(1,1,1)
        plt.hist(s_len_list,  bins=30)
        plt.title("Sequence length distribution")
        plt.ylabel("Counts")
        plt.xlabel("Length range")
        plt.show()
    

def load_rna_ss_dataset(PLT_LOGS=False):
    
    npz_path   = DATASET_PATH["PDB_npz"]
    dataset    = np.load(npz_path)
    
    print("\nLoad npz type dataset: {}".format(npz_path))
    print("- keys: {}".format(dataset.files))
    
    X = dataset['X'] # sequence features
    A = dataset['A'] # backbone adjacency matrix
    Y = dataset['Y'] # original rna-ss matrix
    L = dataset['L'] # sequence length
    
    print("- (X) sequence  features shape: {}".format(X.shape))
    print("- (A) backbone  features shape: {}".format(A.shape))
    print("- (Y) structure features shape: {}".format(Y.shape))
    print("- (L) sequence  length   shape: {}".format(L.shape))
    
    if PLT_LOGS:
        plt.figure(figsize=(10,5))
        plt.subplot(1,1,1)
        plt.hist(L,  bins=30)
        plt.title("Sequence length distribution")
        plt.ylabel("Counts")
        plt.xlabel("Length range")
        plt.show()
    
    return X, A, Y, L


def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape