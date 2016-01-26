"""
    utils
    ~~~~~~~~~~
    The module contains the following functions that are needed for the algorithms
    module:
    1. generate_viable_partition: produces a partition of the graph into a set of
    query nodes and a set of target nodes such that the set of query nodes is a SC graph
    2. compute_fundamental: computes the fundamental matrix from a transition matrix of a graph
    3. update_fundamental_mat: Implements Sherman-Morisson formula for greedy descent algorithm
    4. update_rev_fundamental_mat: Implements Sherman-Morisson formula for reverse
    greedy descent algorithm
    
    :copyright: 2016 Igor Trpevski, <itrpevski@manu.edu.mk> 
    :license: GNU General Public License
"""
from __future__ import division
import networkx as nx
from numpy import array, zeros
from numpy.random import RandomState
from scipy.sparse import csc_matrix, eye
from scipy.sparse.linalg import inv


def choose_random_nodes(G, ntr = 1, n_edges = 1):
    '''
    Returns a random set of absorbing nodes, if the other nodes in the
    graph form a connected component after removing the absorbing nodes.
    Otherwise it returns None
    Parameters
    ----------
    G : Networkx graph
        The graph from which the team will be selected.
    ntr : the number of absorbing nodes
    Returns
    -------
    
    nodes_to_remove : 
        The list of nodes in t the graph to be made absorbing.
    is_viable:
        Boolean indicating whether the graph will stay connected after making
        the nodes absorbing, meaning wheather the partition is viable
    '''
    prng = RandomState()
    order = array(G.nodes())
    nodes_to_remove = list(prng.choice(order, ntr, replace=False ))
    
    H = G.copy()
    H.remove_nodes_from(nodes_to_remove)
    if G.is_directed():
        n_components = nx.number_strongly_connected_components(H)
    else:
        n_components =  nx.number_connected_components(H)
        
    if n_components == 1:
        is_viable = True
        if G.is_directed():
            for node in nodes_to_remove:
                if (H.number_of_nodes() - len(set(G.predecessors(node)) 
                                           - set(nodes_to_remove))) < n_edges:
                    is_viable = False
                    break  
        else:
            for node in nodes_to_remove:
                if (H.number_of_nodes() - len(set(G.neighbors(node)) 
                                           - set(nodes_to_remove))) < n_edges:
                    is_viable = False
                    break   
        return nodes_to_remove, is_viable
        
    else:
        is_viable = False
        return nodes_to_remove, is_viable
    
    
def update_fundamental_mat(F, G, map_node_order, source_node):
    '''
    TODO: to write here
    '''
    n_F = F.shape[0]
    v = zeros((1, n_F))
    deg = G.degree(source_node)
    idx = [map_node_order[i] for i in G.neighbors(source_node) if i in map_node_order.keys()]
    v[:,idx] = 1 / (deg * (deg + 1))
    v = csc_matrix(v)
    u = F[: , map_node_order[source_node]]
    shm_numerator = 1 + v.dot(u)
    F_updated = F - (u.dot(v.dot(F))) / shm_numerator 
    return F_updated


def update_rev_fundamental_mat(F, G, map_node_order, source_node):
    '''
    TODO: to write here
    '''
    n_F = F.shape[0]
    v = zeros((1, n_F))
    deg = G.degree(source_node)
    idx = [map_node_order[i] for i in G.neighbors(source_node) if i in map_node_order.keys()]
    v[:,idx] = - 1 / (deg * (deg - 1))
    v = csc_matrix(v)
    u = F[: , map_node_order[source_node]]
    shm_numerator = 1 + v.dot(u)
    F_updated = F - (u.dot(v.dot(F))) / shm_numerator
    return F_updated 
    
def generate_viable_partition(G, max_trials, target_set_size, n_edges):
    partition_trials = max_trials
    is_viable = False
    while (partition_trials > 0) != (is_viable):
        nodes_to_remove, is_viable = choose_random_nodes(G, 
                                                         target_set_size, 
                                                         n_edges)
        partition_trials -= 1
#         if is_viable:
            # we obtained a nice partition
#             break
    # TODO resolve this in a more elegant way
    if (partition_trials == 0):
        print "A viable partition of the graph could not be generated \
        in {} trials. Reducing target set size by half".format(max_trials)
        target_set_size = round(target_set_size / 2)
        nodes_to_remove, target_set_size = generate_viable_partition(G,  
                        max_trials, target_set_size, n_edges)    
    return nodes_to_remove, target_set_size

def compute_fundamental(P):
    """
    """
    n = P.shape[0]
    F = inv(eye(n, format='csc') - P.tocsc()).todense()
    return F    


    
