# IDEA AND ORIGINAL CODE WRITTING BY --- mikhail.pogorelyy@stjude.org (Mikhail Pogorelyy)
# CODE REVIEW July 2, 2024 --- kmayerbl@fredhutch.org (Koshlan Mayer-Blackwell)
    # Notes: 
    # * Added python class for UI
    # * added sorting by dist to return sorted indices and sorted distances
    # * and modified behaivior to accomodate different modes cpu or cuda

import numpy as np
import pandas as pd
import time
from tqdm import tqdm
from pathlib import Path
import os 

current_dir = Path(__file__).parent

modes = ['apple_silicon', 'cuda', 'cpu']

class TCRgpu:
    """
    A class used to process TCR sequences and compute nearest neighbors using different modes (CPU, CUDA, Apple Silicon).

    Example usage:
    -------------
    
    from tcrdistgpu.distance import TCRgpu
    import pandas as pd

    tst_tcr = pd.read_csv("data/tmp_tcr.tsv", sep="\t").head(2000)
    tg = TCRgpu(tcrs=tst_tcr, mode='cpu', kbest=10)
    tg.encode_tcrs()
    print(tg.encoded.shape)
    
    sorted_indices, sorted_smallest_k_values = tg.compute()
    
    tg.sanity_test_nn_seqs(i=0, max_dist=150)
    """
    def __init__(self, 
            tcrs = None, 
            tcrs2 = None,
            mode = "cuda", 
            kbest = 10, 
            chunk_size = 1000):
        """
        Initialize the TCRgpu object.

        Parameters:
        -----------
        tcrs : DataFrame, optional
            DataFrame containing TCR sequences.
        mode : str, optional
            Mode of computation, must be one of 'apple_silicon', 'cuda', or 'cpu'. Default is 'cuda'.
        kbest : int, optional
            Number of nearest neighbors to compute. Default is 10.
        chunk_size : int, optional
            Size of chunks for processing. Default is 1000.
        """
        self.tcrs = tcrs
        self.tcrs2 = tcrs2
        
        self.mode = mode 
        self.kbest = kbest
        self.chunk_size = chunk_size

        self.cdr3a_col = 'cdr3a'
        self.cdr3b_col = 'cdr3b'
        self.va_col = 'va'
        self.vb_col = 'vb'
        self.target_length = 29
        
        self.params_vec = self.load_params_vec()
        self.submat = self.load_substitution_matrix()
        
    def load_params_vec(self):
        params_df = pd.read_csv(os.path.join(current_dir,"data/params_v2.tsv"), sep="\t", header=None, names=["feature", "value"])
        params_vec = dict(zip(params_df["feature"], params_df["value"]))
        return params_vec
    def load_substitution_matrix(self):
        submat = np.array(np.loadtxt(os.path.join(current_dir,'data/TCRdist_matrix_mega.tsv'), delimiter='\t', dtype=np.int16))
        return submat
    def load_tst_tcr(self):
        tst_tcr = pd.read_csv(os.path.join(current_dir,"data/tmp_tcr.tsv"), sep="\t")
        return tst_tcr
    def pad_center(self, seq, target_length): # function to pad center with gaps until target length
        """
        Pad a sequence to the target length by adding gaps in the center.

        Parameters:
        -----------
        seq : list
            List of characters representing the sequence.
        target_length : int
            Desired length of the padded sequence.

        Returns:
        --------
        list
            Padded sequence.
        """
        seq_length = len(seq)
        if seq_length >= target_length:
           return seq[:target_length]
        else:
           total_padding = target_length - seq_length
           first_half = seq[:seq_length // 2]
           second_half = seq[seq_length // 2:]
           return first_half + ['_'] * total_padding + second_half


    def encode_tcrs_b(self, tcrs = None):
            """
            Encode TCR sequences (BETA ONLY) using the parameters vector.

            Parameters:
            -----------
            tcrs : DataFrame, optional
                DataFrame containing TCR sequences. If None, uses self.tcrs.

            Returns:
            --------
            ndarray
                Encoded TCR sequences as a numpy array.
            """
            if tcrs is None:
                tcrs = self.tcrs
            #cdr3amat = np.array([self.pad_center(seq = list(seq), target_length=self.target_length ) for seq in self.tcrs[self.cdr3a_col]])
            #cdr3amatint = np.vectorize(self.params_vec.get)(cdr3amat)
            cdr3bmat = np.array([self.pad_center(seq=list(seq), target_length=self.target_length ) for seq in tcrs[self.cdr3b_col]])
            cdr3bmatint = np.vectorize(self.params_vec.get)(cdr3bmat)
            #self.cdr3amatint = cdr3amatint
            self.cdr3bmatint = cdr3bmatint
            cols_to_use = slice(3, -2) #truncate CDR3s
            encoded = np.column_stack([
                np.vectorize(self.params_vec.get)(tcrs[self.vb_col]),
                cdr3bmatint[:,cols_to_use]
            ])
            self.encoded = encoded
            return encoded

    def encode_tcrs_a(self, tcrs = None):
            """
            Encode TCR sequences (BETA ONLY) using the parameters vector.

            Parameters:
            -----------
            tcrs : DataFrame, optional
                DataFrame containing TCR sequences. If None, uses self.tcrs.

            Returns:
            --------
            ndarray
                Encoded TCR sequences as a numpy array.
            """
            if tcrs is None:
                tcrs = self.tcrs
            #cdr3amat = np.array([self.pad_center(seq = list(seq), target_length=self.target_length ) for seq in self.tcrs[self.cdr3a_col]])
            #cdr3amatint = np.vectorize(self.params_vec.get)(cdr3amat)
            cdr3amat = np.array([self.pad_center(seq=list(seq), target_length=self.target_length ) for seq in tcrs[self.cdr3a_col]])
            cdr3amatint = np.vectorize(self.params_vec.get)(cdr3amat)
            #self.cdr3amatint = cdr3amatint
            self.cdr3amatint = cdr3amatint
            cols_to_use = slice(3, -2) #truncate CDR3s
            encoded = np.column_stack([
                np.vectorize(self.params_vec.get)(tcrs[self.va_col]),
                cdr3amatint[:,cols_to_use]
            ])
            self.encoded = encoded
            return encoded


    def encode_tcrs(self, tcrs = None):
        """
        Encode TCR sequences using the parameters vector.

        Parameters:
        -----------
        tcrs : DataFrame, optional
            DataFrame containing TCR sequences. If None, uses self.tcrs.

        Returns:
        --------
        ndarray
            Encoded TCR sequences as a numpy array.
        """
        if tcrs is None:
            tcrs = self.tcrs
        cdr3amat = np.array([self.pad_center(seq = list(seq), target_length=self.target_length ) for seq in tcrs[self.cdr3a_col]])
        cdr3amatint = np.vectorize(self.params_vec.get)(cdr3amat)
        cdr3bmat = np.array([self.pad_center(seq=list(seq), target_length=self.target_length ) for seq in tcrs[self.cdr3b_col]])
        cdr3bmatint = np.vectorize(self.params_vec.get)(cdr3bmat)
        self.cdr3amatint = cdr3amatint
        self.cdr3bmatint = cdr3bmatint
        cols_to_use = slice(3, -2) #truncate CDR3s
        encoded = np.column_stack([
            np.vectorize(self.params_vec.get)(tcrs[self.va_col]),
            cdr3amatint[:,cols_to_use],
            np.vectorize(self.params_vec.get)(tcrs[self.vb_col]),
            cdr3bmatint[:,cols_to_use]
        ])
        self.encoded = encoded
        return encoded

    def compute(self, encoded1= None, encoded2=None, mode = None, sort = True):
        """
        Compute the nearest neighbors for the TCR sequences.

        Parameters:
        -----------
        mode : str, optional
            Mode of computation, must be one of 'apple_silicon', 'cuda', or 'cpu'. Default is None, which uses self.mode.
        sort : bool, optional
            Whether to sort the results. Default is True.

        Returns:
        --------
        tuple
            Tuple containing two ndarrays: indices of the nearest neighbors and their distances.
        """
        if mode is None:
            mode = self.mode
        if mode == "cpu":
            import numpy as mx
        elif mode == "cuda":
            import cupy as mx 
            self.submat = mx.array(self.submat)
        elif mode == "apple_silicon":
            import mlx.core as mx #use this for apple silicon
        else:
            raise ValueError("Mode must be in {modes}")

        if encoded1 is None:
            encoded1 = self.encoded
        if encoded2 is None:
            encoded2 = self.encoded

        tcrs1=mx.array(encoded1).astype(mx.uint8)    
        tcrs2=mx.array(encoded2).astype(mx.uint8)
        
        start_time = time.time()
        result=mx.zeros((tcrs1.shape[0],self.kbest),dtype=mx.uint32) #initialize result array for indices
        result_dist = mx.zeros((tcrs1.shape[0],self.kbest),dtype=mx.uint32) 

        for ch in tqdm(range(0, tcrs1.shape[0], self.chunk_size)): #we process in chunks across tcr1 to not run out of memory
            chunk_end = min(ch + self.chunk_size, tcrs1.shape[0])
            row_range = slice(ch, chunk_end)
            
            # KMB: THIS TO GET DISTANCES BACK, WHICH WILL LIKELY BE USEFUL
            dists = mx.sum(self.submat[tcrs1[row_range, None, :], tcrs2[ None,:, :]],axis=2)
            
            partitioned_indices = mx.argpartition(dists, kth =self.kbest, axis=1)
            # Get the indices of the smallest k elements in each row
            smallest_k_indices  = partitioned_indices[:,:self.kbest]
            # Retrieve the values from arr_2d corresponding to smallest_k_indices
            smallest_k_values   = dists[mx.arange(dists.shape[0])[:, mx.newaxis], smallest_k_indices]
            # Sort both smallest_k_indices and smallest_k_values based on values in smallest_k_values
            sorted_indices      = [mx.argsort(smallest_k_values, axis=1)]
            sorted_orig_indices      = smallest_k_indices[mx.arange(dists.shape[0])[:, mx.newaxis], sorted_indices ]
            sorted_smallest_k_values = mx.sort(smallest_k_values, axis=1)

            result[row_range,:]= sorted_orig_indices
            result_dist[row_range,:]= sorted_smallest_k_values

        end_time = time.time()
        print(f"MODE: {mode} -- {end_time - start_time:.6f} seconds") 
        self.result = result
        self.result_dist = result_dist
        return result, result_dist

    def sanity_test_nn_seqs(self, i, tcrs = None, tcrs2= None, max_dist = 150, mode= None):
        """
        View nearest neighbor sequences for a given TCR.

        Parameters:
        -----------
        i : int
            Index of the TCR sequence to find neighbors for.
        max_dist : int, optional
            Maximum distance to consider for nearest neighbors. Default is 150.
        mode : str, optional
            Mode of computation, must be one of 'apple_silicon', 'cuda', or 'cpu'. Default is None, which uses self.mode.

        Returns:
        --------
        None
        """
        if mode is None:
            mode = self.mode
        if mode == "cpu":
            import numpy as mx
            idx = self.result[i]
            dists = self.result_dist[i]
        elif mode == "cuda":
            idx = np.array(self.result[0].get())
            dists =np.array(self.result_dist[0].get())
        elif mode == "apple_silicon":
            import mlx.core as mx #use this for apple silicon
        else:
            raise ValueError("Mode must be in {modes}")
        
        """
        FOR SANITY TESTING
        """
        if tcrs is None:
            tcrs = self.tcrs
        if tcrs2 is None:
            tcrs2 = self.tcrs2

        max_index = np.searchsorted(dists, max_dist, side='right') - 1
        if max_index == 0:
            max_index = 1
        seq_i = tcrs.iloc[i,:].to_list()
        
        for ix, j in enumerate(idx[0:max_index]):
            seq_j = tcrs2.iloc[j,:].to_list()
            print(i,j,dists[ix],seq_i, seq_j)


import numpy as np
import pandas as pd
import os
import importlib.resources
from tcrdistgpu.distance import TCRgpu
from tqdm import tqdm
from scipy.sparse import dok_matrix
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import roc_curve, auc, roc_auc_score, confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
def knn_tcr(tcr_train,
            tcr_test,
            label_train,
            label_test = None,
            chain = "a", mode = "cpu", kbest = 20,
            krange = range(1,21,2),
            adjust_class_weights = True):

  acc_store = list()
  auc_store = list()
  probs_store  = list()
  k_store = list()

  nrow = tcr_test.shape[0]
  ncol = tcr_train.shape[0]

  tg = TCRgpu(tcrs = tcr_train,
              tcrs2 = tcr_test,
              mode = mode,
              kbest = kbest)

  print(f"--- Encoding TCRs as vectors")
  if chain == "a":
    encoded1 = tg.encode_tcrs_a(tcrs = tcr_test)
    encoded2 = tg.encode_tcrs_a(tcrs = tcr_train)
  if chain == "b":
    encoded1 = tg.encode_tcrs_b(tcrs = tcr_test)
    encoded2 = tg.encode_tcrs_b(tcrs = tcr_train)
  if chain == "ab":
    encoded1 = tg.encode_tcrs(tcrs = tcr_test)
    encoded2 = tg.encode_tcrs(tcrs = tcr_train)

  print(f"--- Computing TCRdistances between query x reference tcrs [({nrow})x({ncol})]")
  indices, distances = tg.compute(encoded1= encoded1, encoded2=encoded2)
  print(f"--- Retained column indices of {kbest} nearest neigbors for each row")
  print(f"--- Shape : {indices.shape}")

  if mode == "cuda":
    indices, distances = indices.get(), distances.get()
  print(f"--- Converting to sparse matrix")
  dok = dok_matrix((nrow, ncol), dtype=np.int16)
  for i, jdx, in tqdm(enumerate(indices), total = nrow):
    jdist = distances[i]
    for j,d in zip(jdx,jdist):
      dok[i,j] = max(1,d)
  print(f"--- Completed sparse matrix {dok.shape}))")

  labels = label_train
  y_test = label_test

  print(f"--- Performing kNN Classification")
  total_k = len([x for x in krange])
  for k in tqdm(krange, total = total_k):
    k_store.append(k)

    # Compute weights using distances and labels
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(label_train), y=label_train)
    class_weight_dict = {cls: weight for cls, weight in zip(np.unique(labels), class_weights)}

    weights = np.zeros_like(distances, dtype = 'float32')
    for i in range(distances.shape[0]):
        for j in range(distances.shape[1]):
            label = labels[indices[i, j]]
            if adjust_class_weights:
              weights[i, j] = class_weight_dict.get(label) / (distances[i, j] + 1e-5)
            else:
              weights[i, j] = 1 / (distances[i, j] + 1e-5)

    # Predict based on weighted votes
    weighted_votes = np.zeros((distances.shape[0], len(np.unique(labels))))
    for i in range(distances.shape[0]):
        for j in range(k):
            label = labels[indices[i, j]]
            weighted_votes[i, label] += weights[i, j]

    # Select the class with the highest weighted vote
    predictions = np.argmax(weighted_votes, axis=1)
    p0 = (weighted_votes[:,0])/ (weighted_votes.sum(axis =1 ))
    p1 = (weighted_votes[:,1])/ (weighted_votes.sum(axis =1 ))
    probs_test = np.column_stack([p0,p1])
    probs_store.append(probs_test)
    # If test data is labeled, we can compute AUC
    if label_test is not None:
      acc_store.append( accuracy_score(label_test, predictions) )
      auc_store.append( roc_auc_score(label_test, probs_test[:,1]))

  if label_test is not None:
      print(f"---Returning AUC metric based on provided labels")
      df = pd.DataFrame({'Accuracy' : acc_store,
                  'AUC': auc_store,
                  'k':k_store,
                  'adjust_class_weights':adjust_class_weights,
                  'mode':mode})
  else:
    print(f"---Returning predictions only")
    df = pd.DataFrame({'k':k_store,
                         'adjust_class_weights':adjust_class_weights,
                         'mode':mode})

  return(df,probs_store )


def sort_out_k(dists, k= 10):
    partitioned_indices = np.argpartition(dists, kth =k, axis=1)
    # Get the indices of the smallest k elements in each row
    smallest_k_indices  = partitioned_indices[:,:k]
    # Retrieve the values from arr_2d corresponding to smallest_k_indices
    smallest_k_values   = dists[np.arange(dists.shape[0])[:, np.newaxis], smallest_k_indices]
    # Sort both smallest_k_indices and smallest_k_values based on values in smallest_k_values
    sorted_indices      = [np.argsort(smallest_k_values, axis=1)]
    sorted_orig_indices      = smallest_k_indices[np.arange(dists.shape[0])[:, np.newaxis], sorted_indices ]
    sorted_smallest_k_values = np.sort(smallest_k_values, axis=1)
    return sorted_orig_indices, sorted_smallest_k_values
