import numpy as np
from scipy.sparse import csr_matrix
import networkx as nx
import matplotlib.pyplot as plt

def cluster_by_distance(dist_matrix, d_threshold, visualize=False):
    """
    Clusters humans based on minimum distance using connected components.
    :param dist_matrix: 2D numpy array of distances between humans.
    :param d_threshold: Threshold distance for connecting humans.
    :param visualize: Boolean to indicate whether to plot the graph.
    :return: Dictionary mapping each cluster to list of human indices.
    """
    # Step 1: Create a binary adjacency matrix based on the threshold
    adjacency_matrix = (dist_matrix < d_threshold).astype(int)
    
    # Step 2: Convert adjacency matrix to sparse matrix format
    sparse_matrix = csr_matrix(adjacency_matrix)
    
    # Step 3: Create a graph from the sparse adjacency matrix
    graph = nx.from_scipy_sparse_array(sparse_matrix)
    
    # Step 4: Find connected components in the graph
    clusters = list(nx.connected_components(graph))
    
    # Step 5: Create a dictionary mapping clusters to human indices
    cluster_dict = {i: list(cluster) for i, cluster in enumerate(clusters)}
    
    # Visualization
    if visualize:
        plt.figure(figsize=(8, 6))
        pos = nx.spring_layout(graph)  # Position nodes using spring layout for better visualization
        nx.draw(graph, pos, with_labels=True, node_size=500, node_color='skyblue', font_size=12, font_weight='bold', edge_color='gray')
        plt.title("Graph Representation of Clusters Based on Minimum Distance")
        plt.show()
    
    return cluster_dict

# Example usage
distance_matrix = np.array([
    [0, 1.2, 3.5, 2.1],
    [1.2, 0, 2.9, 3.6],
    [3.5, 2.9, 0, 1.1],
    [2.1, 3.6, 1.1, 0]
])
d_threshold = 2.0
clusters = cluster_by_distance(distance_matrix, d_threshold, visualize=True)
print("Clusters based on minimum distance:", clusters)
