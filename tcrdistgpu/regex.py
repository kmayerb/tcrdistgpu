"""
From tcrdist/regex.py 

Regex tools for defining regex patterns from a list of seququences, 
aligned using computepal_motif.
"""

import numpy as np
import pandas as pd
from palmotif import compute_pal_motif
import re
from tcrdistgpu.distance import TCRgpu
from scipy.cluster.hierarchy import linkage, to_tree
from scipy.spatial.distance import squareform

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


def extract_hierarchical_groupings(arr, nns_to_loc):
	"""
	Perform hierarchical clustering on a distance matrix and return all groupings at each merge step,
	with leaf indices mapped back to their original identifiers.

	Parameters
	----------
	arr : numpy.ndarray
		A square distance matrix (n x n) representing pairwise distances between elements.
	nns_to_loc : dict
		Dictionary mapping local index (0-based, corresponding to rows/columns of `arr`) to original node identifiers.

	Returns
	-------
	data_groupings : list of list
		A list where each element is a list of original node identifiers that were merged together at
		a step in the hierarchical clustering tree. These represent internal clusters at various levels.
	"""

	condensed = squareform(arr)
	Z = linkage(condensed, method='average')

	# Build the tree and get all nodes
	tree, all_nodes = to_tree(Z, rd=True)

	def get_leaves(node):
		"""Recursively collect leaf node indices under the given node."""
		if node.is_leaf():
			return [node.id]
		return get_leaves(node.left) + get_leaves(node.right)

	# Extract groupings from internal nodes
	groupings = []
	for node in all_nodes:
		if not node.is_leaf():
			groupings.append(get_leaves(node))

	data_groupings= [[nns_to_loc.get(i) for i in group] for group in groupings]
	return data_groupings

def get_local_distances_groupings(data, 
	node, 
	pt_to_node, 
	node_to_pt, 
	cdr3_col = 'cdr3_b_aa',
	v_col = 'v_b_gene', 
	get_leaves = False):
	"""
	For a given TCR node, identify its local neighborhood based on partitioning and compute
	hierarchical groupings of CDR3 sequences using TCRdist.

	Parameters
	----------
	data : pd.DataFrame
		DataFrame containing TCR repertoire data, including CDR3 sequences and V gene annotations.
	node : int
		Index of the focal TCR (centroid node).
	pt_to_node : dict
		Dictionary mapping partition ID to a list of node indices in that partition.
	node_to_pt : dict
		Dictionary mapping each node index to its partition ID.
	cdr3_col : str, default 'cdr3_b_aa'
		Name of the column containing the CDR3 amino acid sequences.
	v_col : str, default 'v_b_gene'
		Name of the column containing V gene annotations.

	Returns
	-------
	groupings : list of list
		Hierarchical groupings of node indices (from the same partition as the input node),
		derived from clustering on TCRdist pairwise distances.
	"""

	pt_i      = node_to_pt.get(node) # lookup partition i, from node

	community = pt_to_node.get(pt_i) # get all nodes in pt_i

	data_i = data.loc[community] # get only columns for that community

	tg = TCRgpu(tcrs = data_i,
			mode = "cpu",
			chunk_size = 1000,
			cdr3b_col = cdr3_col,
			vb_col = v_col)

	e   = tg.encode_tcrs_b(data_i)
	arr = tg.compute_array(e,e)

	nns_to_loc = {i: nn for i, nn in enumerate(community)}
	groupings = extract_hierarchical_groupings(arr, nns_to_loc )
	if get_leaves:
		groupings = groupings + [[i] for i in community]
	return groupings


def generate_hierarchical_metaclonotpes(data, 
							 centroids, 
							 pt_to_node, 
							 node_to_pt,
							 binary_column = 'Epitope',
							 cdr3_col = "CDR3",
							 v_col = "V", 
							 j_col = "J", 
							 max_ambiguity = 5,
							 max_ambiguity_small = 0, 
							 get_leaves = False):
	"""
	Generate hierarchical metaclonotype regex patterns based on local CDR3 sequence neighborhoods.
	
	For each centroid node (representing a TCR clonotype), this function identifies its neighborhood
	within a local repertoire partition, clusters the neighborhood hierarchically, and derives regex
	patterns representing subclusters with minimal ambiguity.

	Parameters
	----------
	data : pd.DataFrame
		DataFrame containing TCR sequence and metadata information. Must include columns for CDR3 sequence,
		V gene, J gene, and a binary target variable.
	centroids : dict
		Dictionary mapping a centroid node index to a list of neighboring sequence indices.
	pt_to_node : dict
		Mapping from partition index to a list of node indices belonging to that partition.
	node_to_pt : dict
		Mapping from each node index to its associated partition index.
	binary_column : str, default 'Epitope'
		Column name indicating the binary target variable (e.g., presence/absence of epitope).
	cdr3_col : str, default 'CDR3'
		Column name for CDR3 amino acid sequences.
	v_col : str, default 'V'
		Column name for V gene usage.
	j_col : str, default 'J'
		Column name for J gene usage.
	max_ambiguity : int, default 5
		Maximum allowable degeneracy (ambiguity) in the generated regex pattern for a metaclonotype.
	max_ambiguity_small : int, default 0
		Reserved for future use (not applied in current implementation).

	Returns
	-------
	pd.DataFrame
		A DataFrame containing metaclonotype pattern information. Each row corresponds to a subcluster
		(a regex-defined group of TCRs) and includes the following columns:
			- binvar: the binary label (e.g., epitope presence) of the centroid
			- node: index of the centroid node
			- node_i: unique ID for the subcluster (e.g., "node.subcluster_id")
			- partition: the partition to which the node belongs
			- vfam: parsed TRBV family code
			- V, J: V and J gene usage
			- CDR3: CDR3 sequence of the centroid
			- n1: number of neighbors in the subcluster
			- pattern1: regex pattern representing the subcluster
			- nns_cdr3: pipe-delimited string of all CDR3s in the subcluster
	"""
	import tqdm
	store = list()
	n = len(centroids.keys())
	store_g = {'groupings':dict(),'patterns':dict()}
	for node, nns in tqdm.tqdm(centroids.items(), total =n, desc="Generating Metaclonotype Patterns"):
		
		#print(node, data.iloc[node][['CDR3','Epitope']])
		binvar = data.iloc[node][binary_column]
		
		v = data.iloc[node][v_col]
		vfam = v.replace("TCRB","").replace("TRB","").split('*')[0].split("-")[0]
		if len(vfam) == 2:
			vfam = f"{vfam[0]}0{vfam[1]}"

		j = data.iloc[node][j_col]
		cdr3 = data.iloc[node][cdr3_col]

		pt_i      = node_to_pt.get(node) # lookup partition i, from node
		groupings = get_local_distances_groupings(data, 
			node, 
			pt_to_node, 
			node_to_pt, 
			cdr3_col = cdr3_col ,
			v_col = v_col,
			get_leaves = get_leaves)
		
		store_g['groupings'][node] = groupings

		cnt = 0
		for nns in groupings:
			cnt = cnt + 1

			longest_nns_index = max(nns, key=lambda i: len(data[cdr3_col].loc[i]))
			x = _index_to_matrix(nns, 
				data, 
				col = cdr3_col, 
				centroid_i= longest_nns_index )
			n1 = len(nns)
			pattern1 = _matrix_to_regex(x,
				ntrim = 0, 
				ctrim = 0, 
				max_ambiguity = max_ambiguity )
			store_g['patterns'].setdefault(node, []).append(pattern1)
			node_i = f"{node}.{cnt}"
			nns_cdr3 = "|".join(data[cdr3_col].loc[nns].to_list())
			store.append((binvar, node, node_i, pt_i, vfam, v, j, cdr3, n1, pattern1, nns_cdr3))

	df = pd.DataFrame(store, columns = f'binvar,node,node_i,partition,vfam,{v_col},{j_col},{cdr3_col},n1,pattern1,nns_cdr3'.split(","))
	df.index = df['node_i'].to_list()	
	return df

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
#   """
#   Example
#   -------


#   """
#   dfnode   = clone_df.iloc[ind,].copy()

#   seqs = dfnode[col].to_list()
#   if centroid is None:
#     pwnode   = pwmat[ind,:][:,ind].copy()
#     iloc_idx = pwnode.sum(axis = 0).argmin() 
#     centroid = dfnode[col].to_list()[iloc_idx]
	
#   matrix, stats = compute_pal_motif(seqs = seqs, centroid = centroid)
#   return matrix



# def _index_to_regex_str(ind, 
#             clone_df, 
#             pwmat, 
#             centroid = None,
#             col = 'cdr3_b_aa', 
#             ntrim = 3, 
#             ctrim = 2,  
#             max_ambiguity = 3):
#   """
#   ind : list 
#     iloc row index in clone_df
#   clone_df : pd.DataFrae
#     DataFrame with cdr3 sequence
#   pwmat: np.ndarray
#     Matrix with pairwise inofrmation
#   col : str
#     'cdr3_b_aa', 
#   ntrim : int
#     default 3, 
#   ctrim : int
#     default 2,  
#   max_ambiguity : int
#     default 3, the maximum number of amino acids at a position before a '.' wildcard is applied
#   """
	
#   mat = _index_to_matrix(ind = ind, clone_df= clone_df, pwmat = pwmat, col = col, centroid = centroid )
	
#   regex_str = _matrix_to_regex(matrix = mat, 
#     ntrim = ntrim, 
#     ctrim = ctrim, 
#     max_ambiguity =  max_ambiguity)

#   return(regex_str)
