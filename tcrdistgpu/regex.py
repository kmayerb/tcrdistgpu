"""
From tcrdist/regex.py 

Regex tools for defining regex patterns from a list of seququences, 
aligned using computepal_motif.
"""

import numpy as np
import pandas as pd
from palmotif import compute_pal_motif
import re

def _list_to_regex_component(l, max_ambiguity = 3):
	""" 
	list of str to regex 

	Parameters
	----------

	l : list
		list of strings
	max_ambiguity : int
		default is 3, more than max results in '.' rather than a set []
	
	Example
	-------
	>>> _list_to_regex_component(['A'])
	'A'
	>>> _list_to_regex_component(['A','T','C'])
	'[ATC]'
	>>> _list_to_regex_component(['A','T','C','G'])
	'.'
	>>> _list_to_regex_component(['A','-','C'])
	'[AC]?'
	"""
	if len(l) < 1:
		s = '.?'
	elif len(l) < 2:
		s = l[0]
	elif len(l) <= max_ambiguity:
		s = f"[{''.join(l)}]"
	else:
		s = '.'
	
	if s == '-':
		s = s.replace('-', '.?') 
	elif s.find('-') != -1:
		s = s.replace('-', '') 
		s = f"{s}?"
	return s


def _index_to_matrix(ind, data, col = 'cdr3_b_aa', centroid_i = None):
  dfnode   = data.iloc[ind,].copy()
  seqs     = dfnode[col].to_list()
  centroid = data[col].iloc[centroid_i]
  matrix, stats = compute_pal_motif(seqs = seqs, centroid = centroid)
  return matrix


def _matrix_to_regex(matrix, ntrim = 3, ctrim = 2, max_ambiguity = 3):
	"""
	Example
	-------
	>>> import numpy as np
	>>> import pandas as pd
	>>> arr = np.array([[ 0.        ,  3.32910223,  0.        ,  0.        ,  0.41764668,
	         0.        ,  0.        ,  0.        ],
	       [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
	         0.        ,  3.60160675,  4.5849625 ],
	       [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
	         0.        ,  0.        ,  0.        ],
	       [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
	         4.5849625 ,  0.        ,  0.        ],
	       [ 4.5849625 ,  0.        ,  0.41764668,  0.        ,  0.        ,
	         0.        ,  0.        ,  0.        ],
	       [ 0.        ,  0.        ,  0.        ,  0.        ,  2.66666667,
	         0.        ,  0.        ,  0.        ],
	       [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
	         0.        ,  0.        ,  0.        ],
	       [ 0.        ,  0.        ,  0.        , -0.01922274,  0.        ,
	         0.        ,  0.        ,  0.        ]])
	>>> matrix = pd.DataFrame(arr, index = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G'], columns = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G'])
	>>> matrix_to_regex(matrix)
	'(G[AQ]D)'
	"""
	regex_parts =[matrix.iloc[:,][matrix.iloc[:,i] != 0].index.to_list() for i in range(ntrim ,matrix.shape[1] - ctrim)]
	regex_parts_str = [_list_to_regex_component(x, max_ambiguity = max_ambiguity) for x in regex_parts] 
	regex_pattern = f"({''.join(regex_parts_str)})"
	return regex_pattern



def _multi_regex(regex , bkgd_cdr3):
	"""
	Search a regex pattern in a list of string 
	"""
	result = [re.search(string = s, pattern = regex ) for s in bkgd_cdr3] 
	result = [1 if (x is not None) else None for x in result]
	return result

def _index_to_seqs(ind, clone_df, col):
	dfnode   = clone_df.iloc[ind,].copy()
	seqs = dfnode[col].to_list()
	return seqs 



def generate_metaclonotypes(data, 
                             centroids, 
                             pt_to_node, 
                             node_to_pt,
                             binary_column = 'Epitope',
                             cdr3_col = "CDR3",
                             v_col = "V", 
                             j_col = "J", 
                             max_ambiguity = 5,
                             max_ambiguity_small = 0):
	"""
	Generate metaclonotype patterns for a set of centroid nodes and their neighborhoods.

	This function computes regex-based CDR3 sequence motifs for each centroid node and its
	neighbors based on three groupings: (1) the immediate neighbors (centroids) from greedy clustering, 
	(2) the full community from Louvain/Leiden/Greedy clustering (pt_to_node), and (3) their intersection.
	It outputs a DataFrame summarizing the derived patterns and associated metadata.

	Parameters
	----------
	data : pd.DataFrame
	    A DataFrame containing TCR sequence information, including CDR3, V, J, and binary labels.
	centroids : dict
	    Dictionary mapping node indices (centroids) to lists of nearest neighbor indices.
	pt_to_node : dict
	    Dictionary mapping partition indices to sets of node indices from Leiden clustering.
	node_to_pt : dict
	    Dictionary mapping node indices to their corresponding partition index.
	binary_column : str, optional
	    Column in `data` representing the binary class label (e.g., 'Epitope'), by default 'Epitope'.
	cdr3_col : str, optional
	    Column in `data` with CDR3 amino acid sequences, by default "CDR3".
	v_col : str, optional
	    Column name for V gene usage, by default "V".
	j_col : str, optional
	    Column name for J gene usage, by default "J".
	max_ambiguity : int, optional
	    Maximum ambiguity allowed when constructing regex patterns for large neighborhoods, by default 5.
	max_ambiguity_small : int, optional
	    Maximum ambiguity allowed when neighborhood size < 4, by default 0.

	Returns
	-------
	pd.DataFrame
	    A DataFrame with one row per centroid, containing:
	    - binary class
	    - node index
	    - Leiden partition index
	    - simplified V-family (vfam)
	    - V and J gene
	    - CDR3 sequence of centroid
	    - neighborhood sizes
	    - regex patterns for each of the three neighborhood types
	    - concatenated CDR3 strings from neighbors

	Notes
	-----
	This function assumes `data` is indexed such that `iloc[node]` refers to the correct row
	for a given node. 

	Dependencies
	------------
	- tqdm for progress display
	- _index_to_matrix: user-defined function to extract aligned CDR3 matrices
	- _matrix_to_regex: user-defined function to derive regex from sequence matrices
	"""
  import tqdm
  store = list()
  n = len(centroids.keys())
  for node, nns in tqdm.tqdm(centroids.items(), total =n, desc="Generating Metaclonotype Patterns"):
    
    #print(node, data.iloc[node][['CDR3','Epitope']])
    binvar = data.iloc[node][binary_column]
    
    v = data.iloc[node][v_col]
    vfam = v.replace("TCRB","").replace("TRB","").split('*')[0].split("-")[0]
    if len(vfam) == 2:
      vfam = f"{vfam[0]}0{vfam[1]}"

    j = data.iloc[node][j_col]
    cdr3 = data.iloc[node][cdr3_col]

    #<pt_i> parititon index
    pt_i = node_to_pt.get(node)
    #<full_community>
    full_com = pt_to_node.get(pt_i)
    #<community/greedy intersection>
    nns_intersection = list(set(full_com).intersection(nns))
    
    x = _index_to_matrix(nns, data, col = cdr3_col, centroid_i= node )
    x2 = _index_to_matrix(full_com, data, col = cdr3_col, centroid_i= node )
    x3 = _index_to_matrix(nns_intersection, data, col = cdr3_col, centroid_i= node )
    
    if len(nns) < 4:
      m = max_ambiguity_small 
    else:
      m = max_ambiguity

    pattern1 = _matrix_to_regex(x, ntrim = 0, ctrim = 0, max_ambiguity = m)
    
    if len(full_com) < 4:
      m = max_ambiguity_small 
    else:
      m = max_ambiguity
      
    pattern2 = _matrix_to_regex(x2, ntrim = 0, ctrim = 0, max_ambiguity = m)
    
    if len(nns_intersection) < 4:
      m= max_ambiguity_small 
    else:
      m = max_ambiguity

    pattern3 = _matrix_to_regex(x3, ntrim = 0, ctrim = 0, max_ambiguity = m)

    nns_cdr3 =  "|".join(data[cdr3_col].iloc[nns].to_list())
    com_cdr3 =  "|".join(data[cdr3_col].iloc[full_com].to_list())
    n1 = len(nns)
    n2 = len(full_com)
    store.append((binvar,node,pt_i, vfam, v, j, cdr3, n1, n2, pattern1, pattern2, pattern3, nns_cdr3))
    
  df = pd.DataFrame(store, columns = f'binvar,node,partition,vfam,{v_col},{j_col},{cdr3_col},n1,n2,pattern1,pattern2,pattern3,nns_cdr3'.split(","))
  df.index = df['node'].to_list()
  #df.sort_values('n1', ascending = False).head(30)  
  return df



# def _index_to_matrix(ind, clone_df, pwmat = None, col = 'cdr3_b_aa', centroid = None ):
# 	"""
# 	Example
# 	-------


# 	"""
# 	dfnode   = clone_df.iloc[ind,].copy()

# 	seqs = dfnode[col].to_list()
# 	if centroid is None:
# 		pwnode   = pwmat[ind,:][:,ind].copy()
# 		iloc_idx = pwnode.sum(axis = 0).argmin() 
# 		centroid = dfnode[col].to_list()[iloc_idx]
	
# 	matrix, stats = compute_pal_motif(seqs = seqs, centroid = centroid)
# 	return matrix



# def _index_to_regex_str(ind, 
# 						clone_df, 
# 						pwmat, 
# 						centroid = None,
# 						col = 'cdr3_b_aa', 
# 						ntrim = 3, 
# 						ctrim = 2,  
# 						max_ambiguity = 3):
# 	"""
# 	ind : list 
# 		iloc row index in clone_df
# 	clone_df : pd.DataFrae
# 		DataFrame with cdr3 sequence
# 	pwmat: np.ndarray
# 		Matrix with pairwise inofrmation
# 	col : str
# 		'cdr3_b_aa', 
# 	ntrim : int
# 		default 3, 
# 	ctrim : int
# 		default 2,  
# 	max_ambiguity : int
# 		default 3, the maximum number of amino acids at a position before a '.' wildcard is applied
# 	"""
	
# 	mat = _index_to_matrix(ind = ind, clone_df= clone_df, pwmat = pwmat, col = col, centroid = centroid )
	
# 	regex_str = _matrix_to_regex(matrix = mat, 
# 		ntrim = ntrim, 
# 		ctrim = ctrim, 
# 		max_ambiguity =  max_ambiguity)

# 	return(regex_str)
