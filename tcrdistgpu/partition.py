"""
"""
def run_louvain(G, resolution = 1, random_state =1 ):
	import networkx as nx
	from community import community_louvain
	node_to_cluster = community_louvain.best_partition(G, 
		random_state = random_state, 
		resolution = resolution)
	return node_to_cluster
	pt_to_node, node_to_pt = get_pt(node_to_cluster)
	centroids = None
	return pt_to_node, node_to_pt, centroids

def run_clump(G, cpus = 2):
	from tcrdistgpu.clump import clump_graph_expensive_by_component_parmap
	cg_exp1 = clump_graph_expensive_by_component_parmap(G, cpus = cpus, min_degree = 1)
	node_to_cluster = dict()
	for k,v in cg_exp1.items():
		for i in v:
			node_to_cluster[i] = k
	pt_to_node, node_to_pt = get_pt(node_to_cluster)
	centroids = cg_exp1
	return pt_to_node, node_to_pt, centroids

def run_leiden(G ,mode = "ModularityVertexPartition", **kwargs):
	import networkx as nx
	import igraph as ig
	import leidenalg
	g, reverse_mapping = nx_to_igraph(G)
	# run Leiden clustering
	if mode == "ModularityVertexPartition":
		partition = leidenalg.find_partition(g, leidenalg.ModularityVertexPartition, **kwargs)
	elif mode == "SurpriseVertexPartition":
		partition = leidenalg.find_partition(g, leidenalg.SurpriseVertexPartition, **kwargs)
	else:
		partition = leidenalg.find_partition(g, leidenalg.CPMVertexPartition, **kwargs)
	# convert to {node: cluster_id} dictionary
	node_to_cluster = {}
	for cluster_id, cluster_nodes in enumerate(partition):
			for node in cluster_nodes:
					original_node = reverse_mapping[node]
					node_to_cluster[original_node] = cluster_id
	pt_to_node, node_to_pt = get_pt(node_to_cluster)
	centroids = None
	return pt_to_node, node_to_pt, centroids

def assign_cluster_to_data(data, node_to_pt, col = "cluster"):
	data = data.copy()
	data[col] = pd.Series(data.index).apply(lambda x: node_to_pt.get(x,1000))
	return data

def get_pt(partition):
	"""
	Assuming we have a partition dictionary where each node points to 
	specific cluster. We want a data structure 
	
	We want to rank partitions by partion size, then produce
	
	pt_to_node - partition to node
	node_to_pt - node to partition
	pt_to_nodes - partition to full list of nodes

	"""
	pt_to_node = dict()
	for k,v in partition.items():
		pt_to_node.setdefault(v,list()).append(k)
	pt_to_node_list = list()
	for k,v in pt_to_node.items():
		pt_to_node_list.append(v)
	pt_to_node_list =sorted(pt_to_node_list, key=lambda k: len(k), reverse=True)
	pt_to_node = dict()
	node_to_pt = dict()
	for i,x in enumerate( pt_to_node_list):
		pt_to_node[i] = list()
		for j in x:
			node_to_pt[j] = i
			pt_to_node[i].append(j)
	return pt_to_node, node_to_pt


def nx_to_igraph(G):
	mapping = dict(zip(G.nodes(), range(len(G.nodes()))))
	reverse_mapping = {v: k for k, v in mapping.items()}
	edges = [(mapping[u], mapping[v]) for u, v in G.edges()]
	g = ig.Graph(edges=edges, directed=False)
	return g, reverse_mapping
