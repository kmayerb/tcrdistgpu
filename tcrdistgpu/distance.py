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
from scipy.sparse import dok_matrix

current_dir = Path(__file__).parent
#current_dir = '/fh/fast/gilbert_p/kmayerbl/TCRdist_GPU/tcrdistgpu'
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
            chunk_size = 1000, 
            cdr3a_col = 'cdr3a',
            cdr3b_col = 'cdr3b',
            va_col = 'va',
            vb_col = 'vb',
            kbest = 20):
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
        
        self.chunk_size = chunk_size

        self.cdr3a_col = cdr3a_col
        self.cdr3b_col = cdr3b_col 
        self.va_col = va_col
        self.vb_col = vb_col 
        self.kbest = kbest
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
           #tcrdist3_center = (seq_length // 2) + 1 # I DID THIS TO MAKE CUT POINT MORE SIMILAR TO TCRDIST3, but it led to more discordant results
           #tcrdist3_center = (seq_length +1) // 2
           tcrdist3_center = (seq_length) // 2
           first_half  = seq[:tcrdist3_center]
           second_half = seq[tcrdist3_center:]
           return first_half + ['_'] * total_padding + second_half

    def encode_vb_only(self, tcrs = None):
        if tcrs is None:
            tcrs = self.tcrs
        encoded = np.column_stack([np.vectorize(self.params_vec.get)(tcrs[self.vb_col])])
        return encoded 

    def encode_va_only(self, tcrs = None):
        if tcrs is None:
            tcrs = self.tcrs
        encoded = np.column_stack([np.vectorize(self.params_vec.get)(tcrs[self.va_col])])
        return encoded 

    def encode_cdr3b_only(self, tcrs = None, cols_to_use = slice(3, -2)):
        cdr3bmat = np.array([self.pad_center(seq=list(seq), target_length=self.target_length ) for seq in tcrs[self.cdr3b_col]])
        cdr3bmatint = np.vectorize(self.params_vec.get)(cdr3bmat)
        #cols_to_use = slice(3, -2) #truncate CDR3s
        if tcrs is None:
            tcrs = self.tcrs
        encoded = np.column_stack([cdr3bmatint[:,cols_to_use]])
        return encoded 

    def encode_cdr3a_only(self, tcrs = None, cols_to_use = slice(3, -2)):
        cdr3amat = np.array([self.pad_center(seq=list(seq), target_length=self.target_length ) for seq in tcrs[self.cdr3a_col]])
        cdr3amatint = np.vectorize(self.params_vec.get)(cdr3amat)
        #cols_to_use = slice(3, -2) #truncate CDR3s
        if tcrs is None:
            tcrs = self.tcrs
        encoded = np.column_stack([cdr3amatint[:,cols_to_use]])
        return encoded 
        
    def encode_tcrs_b(self, tcrs = None, cols_to_use = slice(3, -2)):
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
            encoded = np.column_stack([
                np.vectorize(self.params_vec.get)(tcrs[self.vb_col]),
                #cdr3bmatint[:,]
                cdr3bmatint[:,cols_to_use]
            ])
            self.encoded = encoded
            return encoded

    def encode_tcrs_a(self, tcrs = None, cols_to_use = slice(3, -2)):
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
            #cols_to_use = slice(3, -2) #truncate CDR3s
            encoded = np.column_stack([
                np.vectorize(self.params_vec.get)(tcrs[self.va_col]),
                cdr3amatint[:,cols_to_use]
            ])
            self.encoded = encoded
            return encoded


    def encode_tcrs(self, tcrs = None, cols_to_use = slice(3, -2)):
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
        #cols_to_use = slice(3, -2) #truncate CDR3s
        encoded = np.column_stack([
            np.vectorize(self.params_vec.get)(tcrs[self.va_col]),
            cdr3amatint[:,cols_to_use],
            np.vectorize(self.params_vec.get)(tcrs[self.vb_col]),
            cdr3bmatint[:,cols_to_use]
        ])
        self.encoded = encoded
        return encoded






    def compute_distribution(self, encoded1= None, encoded2=None, mode = None, max_k = None, pmf = False, ignore_self = False, bins=np.arange(0, 401, 12)):

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
        if pmf:
            result=mx.zeros((tcrs1.shape[0],len(bins)-1),dtype=mx.float32) #initialize result array for indices
        else:
            result=mx.zeros((tcrs1.shape[0],len(bins)-1),dtype=mx.uint32) #initialize result array for indices

        for ch in tqdm(range(0, tcrs1.shape[0], self.chunk_size)): #we process in chunks across tcr1 to not run out of memory
            chunk_end = min(ch + self.chunk_size, tcrs1.shape[0])
            row_range = slice(ch, chunk_end)
            
            # KMB: THIS TO GET DISTANCES BACK
            dists = mx.sum(self.submat[tcrs1[row_range, None, :], tcrs2[ None,:, :]],axis=2)

            if max_k is not None:
                partitioned_indices = mx.argpartition(dists, kth = max_k, axis=1)
                # Get the indices of the smallest k elements in each row
                smallest_k_indices  = partitioned_indices[:,:max_k]
                # Retrieve the values from arr_2d corresponding to smallest_k_indices
                smallest_k_values   = dists[mx.arange(dists.shape[0])[:, mx.newaxis], smallest_k_indices]

                dists = smallest_k_values

            # if mode == "cuda":
            #     dists = dists.get()

            if pmf:
                if ignore_self:

                    hist_matrix = mx.apply_along_axis(compute_pmf_ignore_self, 1, dists, bins) 
                else:
                    hist_matrix = mx.apply_along_axis(compute_pmf, 1, dists, bins)   
            else:
                if ignore_self:
                    
                    hist_matrix = mx.apply_along_axis(compute_hist_ignore_self, 1, dists, bins) 
                else:
                    hist_matrix = mx.apply_along_axis(compute_hist, 1, dists, bins)   

            result[row_range,:]= hist_matrix

        end_time = time.time()
        print(f"MODE: {mode} -- {end_time - start_time:.6f} seconds") 
        return result

    def compute(self, encoded1= None, encoded2=None, mode = None, max_k = 20, sort = True):
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
        # TODO ADD calculate_chunk_size() to ensure chunksize is appropriate
        if max_k is not None:
            self.kbest = max_k

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
            
            # KMB: THIS TO GET DISTANCES BACK
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

    def tcrdist_csr(self, data, data2=None, chain= "b", organism = "human", mode = None, max_k = None, max_dist = None):
        """AUTOMATED"""
        chain_names = {'a':'alpha chain','b':'beta chain','ab':'alpha-beta chains'}
        if not chain in chain_names.keys():
            raise ValueError
        if chain == "a":
            print(f"Encoding Data based on {chain_names.get(chain)} only")
            e_data = self.encode_tcrs_a(data)
        elif chain == "b":
            print(f"Encoding Data based on {chain_names.get(chain)} only")
            e_data = self.encode_tcrs_b(data)
        elif chain == "ab":
            print(f"Encoding Data based on {chain_names.get(chain)}")
            e_data = self.encode_tcrs(data)
        else:
            raise ValueError
        if data2 is not None:
            if chain == "a":
                e_data2 = self.encode_tcrs_a(data2)
            elif chain == "b":
                e_data2 = self.encode_tcrs_b(data2)
            elif chain == "ab":
                e_data2 = self.encode_tcrs(data2)
            else:
                raise ValueError
        if data2 is not None:        
            x = self.compute_csr(encoded1= e_data, encoded2= e_data2, mode = mode, max_k = max_k, max_dist = max_dist)
        else:
            x = self.compute_csr(encoded1= e_data, encoded2= e_data, mode = mode, max_k = max_k, max_dist = max_dist)
        return(x)


    def compute_csr(self, encoded1= None, encoded2=None, mode = None, max_k = None, max_dist = None):
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
        # TODO ADD calculate_chunk_size() to ensure chunksize is appropriate
        
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
        
        nrow = tcrs1.shape[0]
        ncol = tcrs2.shape[0]
        
        start_time = time.time()
        dok_mat = dok_matrix((nrow, ncol), dtype = 'int16')
        for ch in tqdm(range(0, tcrs1.shape[0], self.chunk_size)): #we process in chunks across tcr1 to not run out of memory
            chunk_end = min(ch + self.chunk_size, tcrs1.shape[0])
            row_range = slice(ch, chunk_end)

            
            dists = mx.sum(self.submat[tcrs1[row_range, None, :], tcrs2[ None,:, :]],axis=2)



                
            
            # map i (global row indice) to ix (index in the chunk)
            map_ix_to_i = {ix:i for ix,i in enumerate(range(ch, chunk_end))}
            

            if max_dist is not None:
                if mode == "cuda":
                    dists = dists.get()
                ixs, js = np.nonzero(dists <= max_dist)

                orig_indices = [map_ix_to_i.get(ix) for ix in ixs]
                for ix,j in zip(ixs, js):
                    dok_mat[map_ix_to_i.get(ix),j] = max(1, dists[ix,j])

                    

            elif max_k is not None:

                partitioned_indices = mx.argpartition(dists, kth =max_k, axis=1)
                # Get the indices of the smallest k elements in each row
                smallest_k_indices  = partitioned_indices[:,:max_k]
                # Retrieve the values from arr_2d corresponding to smallest_k_indices
                smallest_k_values   = dists[mx.arange(dists.shape[0])[:, mx.newaxis], smallest_k_indices]
                # Sort both smallest_k_indices and smallest_k_values based on values in smallest_k_values
                sorted_indices      = [mx.argsort(smallest_k_values, axis=1)]
                sorted_orig_indices      = smallest_k_indices[mx.arange(dists.shape[0])[:, mx.newaxis], sorted_indices ]
                sorted_smallest_k_values = mx.sort(smallest_k_values, axis=1)
                
                if mode == "cuda":
                    dists = dists.get()
                    sorted_orig_indices = sorted_orig_indices.get()
                for ix , i in enumerate(range(ch, chunk_end)):
                    #import pdb; pdb.set_trace()
                    for j in sorted_orig_indices[0,ix,:]:
                        dok_mat[i, j] = max(1,dists[ix, j])

            else:
                for ix , i in enumerate(range(ch, chunk_end)):
                    #import pdb; pdb.set_trace()
                    for j in range(ncol):
                        dok_mat[i, j] = dists[ix, j]

        dok_mat = dok_mat.tocsr()
        end_time = time.time()
        print(f"MODE: {mode} -- {end_time - start_time:.6f} seconds") 
        return dok_mat

    def compute_array(self, encoded1= None, encoded2=None, mode = None):
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
        dists = mx.sum(self.submat[tcrs1[:, None, :], tcrs2[ None,:, :]],axis=2)
        end_time = time.time()
        print(f"Array computed without chunks {mode} -- {end_time - start_time:.6f} seconds") 
        return dists


    def test_only_(self, a, b, chain = 'b'):
        """
        THIS IS FOR TESTING ONLY
        tg.test_only(
            a = pd.DataFrame({'cdr3b':['CASAAAGF'],'vb':['TRBV9*01']}), 
            b = pd.DataFrame({'cdr3b':['CASAAAGF'],'vb':['TRBV9*01']}))

        tg.test_only(
            a = pd.DataFrame({'cdr3b':['CASAAAGF'],'vb':['TRBV9*01']}), 
            b = pd.DataFrame({'cdr3b':['CASAGGAGF'],'vb':['TRBV9*01']}))
        """
        if chain == "b":
            encoded_seq  = self.encode_tcrs_b(a)
            encoded_seq2 = self.encode_tcrs_b(b)
            dist = np.sum(self.submat[encoded_seq[:, None, :], encoded_seq2[ None,:, :]],axis=2)
        return dist


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


def compute_pmf(row, bins=np.linspace(0, 500, 51)):
    counts, _ = np.histogram(row, bins=bins)
    return counts / np.sum(counts)

def compute_pmf_ignore_self(row, bins=np.linspace(0, 500, 51)):

    try:
        min_index = np.argmin(row)
        row= np.delete(row, min_index)
    except TypeError:
        min_index = np.argmin(row).get()
        row = np.delete(row.get(), min_index)

    counts, _ = np.histogram(row, bins=bins)
    return counts / np.sum(counts)

def compute_hist(row, bins=np.linspace(0, 500, 51)):
    counts, _ = np.histogram(row, bins=bins)
    return counts 

def compute_hist_ignore_self(row, bins=np.linspace(0, 500, 51)):
    try:
        min_index = np.argmin(row)
        row= np.delete(row, min_index)
    except TypeError:
        min_index = np.argmin(row).get()
        row = np.delete(row.get(), min_index)
        
    counts, _ = np.histogram(row, bins=bins)
    return counts 

def calculate_chunk_size(i: int, max_size = 10000000) -> int:
    # Calculate the maximum chunk size
    chunk_size = max_size // i
    return chunk_size

#i = 1001
#chunk_size = calculate_chunk_size(i)
#print(f"For i = {i}, the chunk size is {chunk_size}.")

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
