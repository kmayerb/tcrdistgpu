from tcrdistgpu.distance import TCRgpu
import pandas as pd
import numpy as np 
import time
from scipy.sparse import dok_matrix
from tqdm import tqdm
from pwseqdist import apply_pairwise_rect
import pwseqdist as pw


def legacy_mode(v1, v2, seqs1, seqs2,  submat , mode ='cpu'):
    if mode == "cpu":
        import numpy as mx
    elif mode == "cuda":
        import cupy as mx 
    elif mode == "apple_silicon":
        import mlx.core as mx #use this for apple silicon
    else:
        raise ValueError("Mode must be in {modes}")
    ks =  {'use_numba': True, 'distance_matrix': pw.matrices.tcr_nb_distance_matrix, 
    'dist_weight': 1, 'gap_penalty':4, 'ntrim':3, 'ctrim':2, 'fixed_gappos':True}

    start_time = time.perf_counter()
    dists = pw.apply_pairwise_rect(metric = pw.metrics.nb_vector_tcrdist, 
                            seqs1  = seqs1, 
                            seqs2  = seqs2, 
                            ncpus  = 1, 
                            uniqify = True, 
                            **ks)

    vdists = mx.sum(submat[v1[:, None, :], v2[ None,:, :]],axis=2)
    
    end_time = time.perf_counter()
    print(f"MODE: {mode} -- {end_time - start_time:.6f} seconds") 
    return (vdists, 3*dists)



def legacy_mode_sparse(v1, v2, seqs1, seqs2,  submat , mode ='cpu', chunk_size = 50, max_k = None, max_dist = None):
    if mode == "cpu":
        import numpy as mx
    elif mode == "cuda":
        import cupy as mx 
    elif mode == "apple_silicon":
        import mlx.core as mx #use this for apple silicon
    else:
        raise ValueError("Mode must be in {modes}")
    ks =  {'use_numba': True, 'distance_matrix': pw.matrices.tcr_nb_distance_matrix, 
    'dist_weight': 1, 'gap_penalty':4, 'ntrim':3, 'ctrim':2, 'fixed_gappos':True}

    start_time = time.perf_counter()

    start_time = time.time()
    nrow = len(seqs1)
    ncol = len(seqs2)

    assert(v1.shape[0] == nrow)
    assert(v2.shape[0] == ncol)

    dok_mat = dok_matrix((nrow, ncol), dtype = 'int16')

    for ch in tqdm(range(0, seqs1.shape[0], chunk_size)): 
        chunk_end = min(ch + chunk_size, nrow)
        row_range = slice(ch, chunk_end)

        dists = pw.apply_pairwise_rect(metric = pw.metrics.nb_vector_tcrdist, 
                                seqs1  = seqs1[row_range], 
                                seqs2  = seqs2, 
                                ncpus  = 1, 
                                uniqify = True, 
                                **ks)

        vdists = mx.sum(submat[v1[row_range, None, :], v2[ None,:, :]],axis=2)
        dists = (vdists + 3*dists)


        # map i (global row indice) to ix (index in the chunk)
        map_ix_to_i = {ix:i for ix,i in enumerate(range(ch, chunk_end))}
        

        if max_dist is not None:
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
            for ix , i in enumerate(range(ch, chunk_end)):
                #import pdb; pdb.set_trace()
                for j in sorted_orig_indices[0,ix,:]:
                    dok_mat[map_ix_to_i.get(ix), j] = max(1,dists[ix, j])
        else:
            for ix , i in enumerate(range(ch, chunk_end)):
                #import pdb; pdb.set_trace()
                for j in range(ncol):
                    dok_mat[map_ix_to_i.get(ix), j] = dists[ix, j]

    dok_mat = dok_mat.tocsr()
    end_time = time.time()
    print(f"MODE: {mode} -- {end_time - start_time:.6f} seconds") 
    return dok_mat



def compute_legacy_chunk(v1, seqs1, v2, seqs2, submat , mode ='cpu', max_k = None, max_dist = None):
    if mode == "cpu":
        import numpy as mx
    elif mode == "cuda":
        import cupy as mx 
    elif mode == "apple_silicon":
        import mlx.core as mx #use this for apple silicon
    else:
        raise ValueError("Mode must be in {modes}")

    ks =  {'use_numba': True, 'distance_matrix': pw.matrices.tcr_nb_distance_matrix, 
    'dist_weight': 1, 'gap_penalty':4, 'ntrim':3, 'ctrim':2, 'fixed_gappos':True}
    


    nrow = len(seqs1)
    ncol = len(seqs2)

    assert(v1.shape[0] == nrow)
    assert(v2.shape[0] == ncol)

    dok_mat = dok_matrix((nrow, ncol), dtype = 'int16')

    dok_mat= dok_matrix 
    dok_mat = dok_matrix((nrow, ncol), dtype = 'int16')

    dists = pw.apply_pairwise_rect(metric = pw.metrics.nb_vector_tcrdist, 
                            seqs1  = seqs1, 
                            seqs2  = seqs2, 
                            ncpus  = 1, 
                            uniqify = True, 
                            **ks)

    vdists = mx.sum(submat[v1[:, None, :], v2[ None,:, :]],axis=2)
    dists = (vdists + 3*dists)


    # map i (global row indice) to ix (index in the chunk)
    ch = 0
    chunk_end = nrow
    map_ix_to_i = {ix:i for ix,i in enumerate(range(ch, chunk_end))}
    

    if max_dist is not None:
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
        for ix , i in enumerate(range(ch, chunk_end)):
            #import pdb; pdb.set_trace()
            for j in sorted_orig_indices[0,ix,:]:
                dok_mat[map_ix_to_i.get(ix), j] = max(1,dists[ix, j])
    else:
        for ix , i in enumerate(range(ch, chunk_end)):
            #import pdb; pdb.set_trace()
            for j in range(ncol):
                dok_mat[map_ix_to_i.get(ix), j] = dists[ix, j]

    dok_mat = dok_mat.tocsr()
    return dok_mat


def compute_legacy_sparse_multicpu(v1, seqs1, v2, seqs2, submat , mode ='cpu', max_k = None, max_dist = None, chunk_size = 50, cpus = 2):
    nrow = len(seqs1)
    chunk_idx = list()
    for ch in tqdm(range(0, seqs1.shape[0], chunk_size)): 
        chunk_end = min(ch + chunk_size, nrow)
        row_range = slice(ch, chunk_end)
        chunk_idx.append(row_range)
    #print(chunk_idx)
    chunked_seqs1 = [seqs1[x] for x in chunk_idx]
    chunked_v1    = [v1[x] for x in chunk_idx]
    import parmap
    import scipy.sparse as sp
    csr_matrices= parmap.starmap(_compute_legacy_chunk, list(zip(chunked_v1, chunked_seqs1)), v2 = v2,  seqs2 = seqs2, submat= submat, 
        max_k = max_k , max_dist = max_dist,
        pm_pbar = True, pm_processes=cpus)
    csr = sp.vstack(csr_matrices)
    return csr




