import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader, TensorDataset, Subset

# Complementary rules; how nucleotides pair with each other 
COMP_DICT = {
    "A": "T",
    "T": "A",
    "G": "C",
    "C": "G"
}

####################################
### DATA LOADING AND PREPARATION ###
####################################

def compl_rev_X(mat, at_per_nt=5, seq_len=6):
    '''
    Get distance matrix of complementary reversed sequence.

    mat -- distance matrix of shape (n, m, m)
    at_per_nt -- number of atoms from each nucleotide used for
    distance calcaulations
    '''
    submat_size = at_per_nt * seq_len   # 6 for hexamer

    # top left
    submat1 = mat[:, :submat_size, :submat_size]

    # top right
    submat2 = mat[:, :submat_size, submat_size:]

    # bottom left
    submat3 = mat[:, submat_size:, :submat_size]

    # bottom right
    submat4 = mat[:, submat_size:, submat_size:]

    hstack1 = np.concatenate([submat4, submat3], axis=2)
    hstack2 = np.concatenate([submat2, submat1], axis=2)

    vstack = np.concatenate([hstack1, hstack2], axis=1)

    return vstack



def compl_rev_seq(seq):
    """
    Converts DNA sequence to its complementary reverse.
    Input: str
    Output: str
    """
    return "".join(COMP_DICT[s] for s in seq[::-1])



def remove_nts(mat, nts, at_per_nt=5):
    """
    Remove structural information about selected nucleotides 
    in hexamer.
    
    mat -- distance matrix/matrices with shape (..., size, size)
    nts -- 1-based nucleotide indices to remove
    at_per_nt -- number of atoms considered in each nucleotide
    """

    mat_size = mat.shape[-1]

    if mat.shape[-2] != mat_size:
        raise ValueError("Distance matrices must be square.")

    n_nts = mat_size // (2 * at_per_nt)

    if any(nt < 1 or nt > n_nts for nt in nts):
        raise ValueError(
            f"1-based nt index must be in range 1-{n_nts}"
        )

    idx_A = [
        (i - 1) * at_per_nt + j
        for j in range(at_per_nt)
        for i in nts
    ]

    idx_B = [mat_size - 1 - i for i in idx_A]

    all_idx = idx_A + idx_B

    mat = np.delete(mat, all_idx, axis=-2)
    mat = np.delete(mat, all_idx, axis=-1)

    return mat



def Train_Test_Indices(index_ranges, mode, train_fraction, seed=42):
    """
    Prepares indices in range: (0 -- number of samples) shuffled and splited 
    into train and test sets. Works in two modes:
    1. random -- indices are splited bewteen train and test completely randomly 
    2. by_hexamer -- blocks of indices corresponding to given hexamer are randomly
    splited between train and test; random split takes place on level of hexamers, 
    not individual samples.

    Input:
    index_ranges -- dictionary with index ranges for all hexamers
    mode -- either 'random' or 'by_hexamer'
    seed -- random seed for reproducibility (default: 42)

    Returns:
    train_indices, test_indices -- shuffled indices of train / test set samples 
    split_info -- only in 'by_hexamer' mode (None in 'random'). Contains information
    which hexamers ended up in train and test sets.
    """
    rng = np.random.default_rng(seed=seed)
    
    def _split_indices(n):
        """
        n - int
        
        Returns indicies in range: 0 -- n splited into two sets: 
        train (size = n * TRAIN_SIZE) and test (size = n * (1 - TRAIN_SIZE))
        """
    
        all_inds = np.arange(0, n)
        rng.shuffle(all_inds)
        n_train = int(n * train_fraction)
        train_inds = all_inds[:n_train]
        test_inds = all_inds[n_train:]
    
        return train_inds, test_inds
    
    if mode == "random":
        n = max(index_ranges.values())[1]
        split_info = None
        train_indices, test_indices = _split_indices(n)
        
        return train_indices, test_indices, split_info
    
    elif mode == "by_hexamer":
        # hexamer and its corresponding complementary reversed 
        # hexamer shouldn't be splited between train and test

        n = len(index_ranges)
        hex_train_indices, hex_test_indices = _split_indices(n)
    
        train_indices = []
        test_indices = []
        split_info = {
            "train": [], 
            "test": []
        }
        
        for i, (k, v) in enumerate(index_ranges.items()):
            if i in hex_train_indices:
                train_indices += np.arange(v[0], v[1]).tolist()
                split_info["train"].append(k)
            else:
                test_indices += np.arange(v[0], v[1]).tolist()
                split_info["test"].append(k)
    
        rng.shuffle(train_indices)
        rng.shuffle(test_indices)

        return np.array(train_indices), np.array(test_indices), split_info

    else:
        raise ValueError("Split mode must be 'random' or 'by_hexamer'.")



def Train_Val_Test_Indices(index_ranges, mode, train_fraction, val_fraction=0.1, seed=42):
    """
    Prepares indices in range: (0 -- number of samples) shuffled and splited 
    into train, test and validation sets. Works in two modes:
    1. random -- indices are splited bewteen train and test completely randomly 
    2. by_hexamer -- blocks of indices corresponding to given hexamer are randomly
    splited between train and test; random split takes place on level of hexamers, 
    not individual samples.

    Input:
    index_ranges -- dictionary with index ranges for all hexamers
    mode -- either 'random' or 'by_hexamer'
    seed -- random seed for reproducibility (default: 42)

    Returns:
    train_indices, test_indices -- shuffled indices of train / test set samples 
    split_info -- only in 'by_hexamer' mode (None in 'random'). Contains information
    which hexamers ended up in train and test sets.
    """
    rng = np.random.default_rng(seed=seed)

    if 1 - train_fraction < val_fraction:
        raise ValueError("Validation fraction can't be larger that 1-train_fraction.")
    elif 1 - train_fraction - val_fraction < 0.1:
        print("Warning! Test set consist 10% or less of the total samples.\
              Reduce train_fraction or val_fraction to increase test set.")
    else:
        pass
    
    def _split_indices(n):
        """
        n - int
        
        Returns indicies in range: 0 -- n splited into two sets: 
        train (size = n * TRAIN_SIZE) and test (size = n * (1 - TRAIN_SIZE))
        """
    
        all_inds = np.arange(0, n)
        rng.shuffle(all_inds)
        n_train = int(n * train_fraction)
        n_train_val = int(n * (train_fraction + val_fraction))
        train_inds = all_inds[:n_train]
        val_inds = all_inds[n_train:n_train_val]
        test_inds = all_inds[n_train_val:]
    
        return train_inds, val_inds, test_inds
    
    if mode == "random":
        n = max(index_ranges.values())[1]
        split_info = None
        train_indices, val_indices, test_indices = _split_indices(n)
        
        return train_indices, val_indices, test_indices, split_info
    
    elif mode == "by_hexamer":
        # hexamer and its corresponding complementary reversed 
        # hexamer shouldn't be splited between train and test

        n = len(index_ranges)
        hex_train_indices, hex_val_indices, hex_test_indices = _split_indices(n)
    
        train_indices = []
        val_indices = []
        test_indices = []
        split_info = {
            "train": [], 
            "validation": [],
            "test": []
        }
        
        for i, (k, v) in enumerate(index_ranges.items()):
            if i in hex_train_indices:
                train_indices += np.arange(v[0], v[1]).tolist()
                split_info["train"].append(k)
            elif i in hex_val_indices:
                val_indices += np.arange(v[0], v[1]).tolist()
                split_info["validation"].append(k)
            else:
                test_indices += np.arange(v[0], v[1]).tolist()
                split_info["test"].append(k)
    
        rng.shuffle(train_indices)
        rng.shuffle(val_indices)
        rng.shuffle(test_indices)

        return np.array(train_indices), np.array(val_indices), np.array(test_indices), split_info

    else:
        raise ValueError("Split mode must be 'random' or 'by_hexamer'.")



def upper_triangle_to_vector(arr):
    """
    Convert an array of shape (n, m, m) into an array of shape (n, m * (m - 1) // 2)
    by extracting the upper triangle of each matrix, excluding the diagonal.

    Parameters
    ----------
    arr : np.ndarray
        Input array of shape (n, m, m).

    Returns
    -------
    np.ndarray
        Array of shape (n, m * (m - 1) // 2).
    """

    rows, cols = np.triu_indices(arr.shape[1], k=1)

    return arr[:, rows, cols]



def Standarize_inplace(x):
    '''
    x -- np.array
    '''
    m = x.mean()
    std = x.std()

    x -= m
    x /= std



class DataSet:
    """
    Object creates and stores train and test data loaders.
    """
    def __init__(self, X, y, train_idx, test_idx, val_idx=None, batch_size=32):
        if isinstance(X, torch.Tensor):
            self.X = [X]
        elif isinstance(X, list):
            if all(isinstance(xi, torch.Tensor) for xi in X):
                self.X = X
            else:
                raise TypeError("X must be torch tensor or list of torch tensors.")
        else:
            raise TypeError("X must be torch tensor or list of torch tensors.")

        if isinstance(y, torch.Tensor):
            self.y = y
        else:
            raise TypeError("y must be torch tensor.")

        self.train_idx = np.asarray(train_idx)
        self.test_idx = np.asarray(test_idx)
        self.val_idx = np.asarray(val_idx) if val_idx is not None else None
        self.batch_size = batch_size

        tensorset_full = TensorDataset(*self.X, self.y)
        tensorset_train = Subset(tensorset_full, self.train_idx)
        tensorset_test = Subset(tensorset_full, self.test_idx)
        if self.val_idx is not None:
            tensorset_val = Subset(tensorset_full, self.val_idx)
        
        self.train_loader = DataLoader(tensorset_train, batch_size=self.batch_size, shuffle=True)
        self.test_loader = DataLoader(tensorset_test, batch_size=self.batch_size, shuffle=False)
        if self.val_idx is not None:
            self.val_loader = DataLoader(tensorset_val, batch_size=self.batch_size, shuffle=False)
        else:
            self.val_loader = None

    def example_inp(self):
        """
        Returns example batched input from training set.
        """
        return next(iter(self.train_loader))



##########################
### ANALYSIS FUNCTIONS ###
##########################

def get_confusion_matrix(df):
    """
    Produces confusion matrix showing how often (in %) different 
    classes were confused with other classes by the model.
    
    Input:
    df:
        Dataframe with predictions. It must contain
        'real' and 'pred' columns.

    Returns:
        Dataframe of size n_classes x n_classes.
    """
    conf_counts = [
        list(i[0]) + [i[1]] for i in 
        df[["real", "pred"]].groupby("real").value_counts().items()
    ]
    conf_df = pd.DataFrame(conf_counts, columns=["r", "p", "c"])
    conf_mat = conf_df.pivot(index='r', columns='p', values='c')
    conf_mat = conf_mat.divide(conf_mat.sum(axis=1), axis=0)

    # check if all columns are present 
    ind_col_diff = list(set(conf_mat.index) - set(conf_mat.columns))
    if ind_col_diff:
        missing = {i:np.where(conf_mat.index == i)[0].item() for i in ind_col_diff}
        sorted_labels = sorted(missing, key=lambda x: missing[x])
        for col in sorted_labels:
            vals = np.zeros(len(conf_mat.index))
            ind = missing[col]
            conf_mat.insert(ind, col, vals)

    return conf_mat.fillna(0.0)



def plt_conf_mat(conf_mat, title=None, out_path=None):
    fig_score, ax_score = plt.subplots(1, 1)
    fig_score.set_size_inches(10, 10)
    ax_score.imshow(conf_mat.values, cmap='RdYlGn')
    for i in range(16):
        for j in range(16):
            ax_score.text(j, i, np.round(conf_mat.iloc[i, j] * 100, 1), ha='center', va='center', fontsize=11)
    ax_score.set_xticks(list(range(16)))
    ax_score.set_yticks(list(range(16)))
    ax_score.xaxis.set_ticks_position('top')
    ax_score.set_xticklabels(conf_mat.index) 
    ax_score.set_yticklabels(conf_mat.columns)
    ax_score.set_ylabel("Class:", fontsize=12)
    ax_score.set_xlabel("Predicted as:", fontsize=12)
    ax_score.xaxis.set_label_position('top')
    if title is not None:
        ax_score.set_title(title, fontsize=12)

    if out_path is not None:
        fig_score.savefig(out_path, dpi=300, facecolor='white', bbox_inches='tight')
        plt.close(fig_score)
    else:
        plt.show()
