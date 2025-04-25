from scipy.spatial.distance import pdist, squareform  
import torch
def calculate_adjacency_matrix(spatiotemporal_tensor):
    """
    Calculates the adjacency matrix based on the correlation of the 'time' dimension of a spatiotemporal tensor
    using 1 - squareform(pdist(..., 'correlation')).

    :param spatiotemporal_tensor: A 2D numpy array of shape [node, time]
    :return: A 2D numpy array representing the adjacency matrix of shape [node, node]
    """ 
    adjacency_matrix = 1 - squareform(pdist(spatiotemporal_tensor, 'correlation'))
    return torch.tensor(adjacency_matrix)