"""
    algorithms
    ~~~~~~~~~~
    The module contains the following algorithms that choose a set of links
    between a set of query nodes and a set of target nodes in a graph.
    absorbing centrality:
    1. greedy_navigation: Implements a greedy descent algorithm
    2. random_links: Randomly chooses links between two sets
    3. exhaustive_set: Exhaustive search algorithm
    4. link_prediction: link prediction algorithms from networkx
    5. reverse_greedy:  Implements a reverse greedy algorthm
    :copyright: 2016 Igor Trpevski, <itrpevski@manu.edu.mk> 
    :license: GNU General Public License
"""

from __future__ import division

import networkx as nx
from itertools import product, combinations
from numpy import zeros, asarray, min, argmin
from numpy.random import RandomState
from random import sample

from devel.utils import (update_fundamental_mat,
                         update_rev_fundamental_mat, 
                         compute_fundamental)

from scipy.sparse.csc import csc_matrix

def greedy_navigation(G, query_nodes, target_nodes, n_edges, start_dist):
    """Selects a set of links with a greedy descent algorithm that reduce the 
    absorbing RW centrality between a set of query nodes Q and a set of absorbing
    target nodes C such that Q \cap C = \emptyset. The query and target set 
    must be a 'viable' partition of the graph.
    Parameters
    ----------
    G : Networkx graph
        The graph from which the team will be selected.
    query : list 
        The set of nodes from which random walker starts.
    target : list
        The set of nodes from where the random walker ends.
    n_edges : integer
        the number of links to be added
    start_dist: list
        The starting distribution over the query set
    P : Scipy matrix
        The transition matrix of the graph G
    F : Scipy matrix
        The fundamental matrix for the graph G with the given set of absorbing
        random walk nodes
    Returns
    -------
    links : list
        The set of links that reduce the absorbing RW centrality
    """
    H = G.copy()
    prng = RandomState()
    query_set_size = len(query_nodes)
    target_set_size = len(target_nodes)
    map_query_to_org = dict(zip(query_nodes, range(query_set_size)))

    P = csc_matrix(nx.google_matrix(H, alpha=1))
    P_abs = P[list(query_nodes),:][:,list(query_nodes)]
    F = compute_fundamental(P_abs)
    row_sums = start_dist.dot(F.sum(axis=1))[0,0]
    best_F = zeros(F.shape)
    optimal_set = []
    ac_scores = []
    ac_scores.append(row_sums)
    
    while n_edges > 0:
        round_min = -1
        best_node = -1
        
        for i in query_nodes:
            abs_neighbours = [l for l in H.neighbors(i) if l in target_nodes]
            if len(abs_neighbours) == target_set_size:
                continue
            
            F_updated = update_fundamental_mat(F, H, map_query_to_org, i)
            abs_cen = start_dist.dot( F_updated.sum(axis = 1))[0,0]
            if abs_cen < round_min or round_min == -1:
                best_node = i
                round_min = abs_cen
                best_F = F_updated
        F = best_F            
        ac_scores.append(round_min)
        optimal_candidate_edges = [(best_node, k, round_min) 
                                   for k in target_nodes 
                                   if H.has_edge(best_node, k) == False ]
        
        try:
            edge_idx = prng.randint(0, len(optimal_candidate_edges))
        except ValueError:
            print(H.neighbors(best_node))
            print([l for l in H.neighbors(best_node) if l in target_nodes])
            print(best_node)
            print(optimal_candidate_edges)
            print(target_nodes)
        H.add_edge(optimal_candidate_edges[edge_idx][0], 
                   optimal_candidate_edges[edge_idx][1])
        optimal_set.append(optimal_candidate_edges[edge_idx])
        n_edges -= 1

    return optimal_set, ac_scores


def random_links(G, query_nodes, target_nodes, n_edges, start_dist):
    """Selects a random set of links between a set of query nodes Q and a set of absorbing
    target nodes C such that Q \cap C = \emptyset. 
    Parameters
    ----------
    G : Networkx graph
        The graph from which the team will be selected.
    query : list 
        The set of nodes from which random walker starts.
    target : list
        The set of nodes from where the random walker ends.
    n_edges : integer
        the number of links to be added
    start_dist: list
        The starting distribution over the query set
    Returns
    -------
    links : list
        The set of links that reduce the absorbing RW centrality
    ac_scores: list
        The set of scores of adding the links
    """
    query_set_size = len(query_nodes)
    map_query_to_org = dict(zip(query_nodes, range(query_set_size)))
    P = csc_matrix(nx.google_matrix(G, alpha=1))
    P_abs = P[list(query_nodes),:][:,list(query_nodes)]
    F = compute_fundamental(P_abs)
    row_sums = start_dist.dot(F.sum())[0,0]
    candidates = list(product(query_nodes, target_nodes))
    eligible = [candidates[i] for i in range(len(candidates)) 
                if G.has_edge(candidates[i][0], candidates[i][1]) == False]
    links_to_add = sample(eligible, n_edges)
    
    ac_scores = []
    ac_scores.append(row_sums)
    i = 0
    while i < n_edges:
        F_updated = update_fundamental_mat(F, G, map_query_to_org, links_to_add[i][0])
        G.add_edge(links_to_add[i][0], links_to_add[i][1])
        abs_cen = start_dist.dot(F_updated.sum(axis = 1))[0,0]
        F = F_updated            
        ac_scores.append(abs_cen)
        i += 1
    return links_to_add, ac_scores

def exhaustive_set(G, query_nodes, target_nodes, n_edges, start_dist):
    """Exaustively searches all the combinations of k links between 
    a set of query nodes Q and a set of absorbing
    target nodes C such that Q \cap C = \emptyset. 
    Parameters
    ----------
    G : Networkx graph
        The graph from which the team will be selected.
    query : list 
        The set of nodes from which random walker starts.
    target : list
        The set of nodes from where the random walker ends.
    n_edges : integer
        the number of links to be added
    start_dist: list
        The starting distribution over the query set
    Returns
    -------
    links : list
        The set of links that reduce the absorbing RW centrality
    ac_scores: list
        The set of scores of adding the links
    """
    query_set_size = len(query_nodes)
    map_query_to_org = dict(zip(query_nodes, range(query_set_size)))
    P = csc_matrix(nx.google_matrix(G, alpha=1))
    P_abs = P[list(query_nodes),:][:,list(query_nodes)]
    F = compute_fundamental(P_abs)
    row_sums = start_dist.dot(F.sum())[0,0]
    candidates = list(product(query_nodes, target_nodes))
    eligible = [candidates[i] for i in range(len(candidates)) 
                if G.has_edge(candidates[i][0], candidates[i][1]) == False]
    ac_scores = [row_sums]
    exhaustive_links = []
    for L in range(1, n_edges+1):
        print '\t Number of edges {}'.format(L)
        round_min = -1
        best_combination = [] 
        for subset in combinations(eligible, L):
            H = G.copy()
            F_modified = F.copy()
            for links_to_add in subset:
                F_updated = update_fundamental_mat(F_modified, H, map_query_to_org, links_to_add[0])
                H.add_edge(links_to_add[0], links_to_add[1])
                F_modified = F_updated            
            abs_cen = start_dist.dot( F_updated.sum(axis = 1))[0,0]
            if abs_cen < round_min or round_min == -1:
                best_combination = subset
                round_min = abs_cen
        exhaustive_links.append(best_combination)
        ac_scores.append(round_min)              
    return exhaustive_links, ac_scores


def link_prediction(G, query_nodes, target_nodes, n_edges, start_dist, alg = "ra"):
    """Selects a random set of links between based on the scores calculated by 
    a standard link-prediction algorithm from networkx library
    Parameters
    ----------
    G : Networkx graph
        The graph from which the team will be selected.
    query : list 
        The set of nodes from which random walker starts.
    target : list
        The set of nodes from where the random walker ends.
    n_edges : integer
        the number of links to be added
    start_dist: list
        The starting distribution over the query set
    alg: string
        A string describing the link-prediction algorithm to be used
    Returns
    -------
    links : list
        The set of links that reduce the absorbing RW centrality
    ac_scores: list
        The set of scores of adding the links
    """
    assert alg in ["ra", "pa", "jaccard", "aa"], "alg must be one of [\"ra\", \"pa\", \"jaccard\", \"aa\"]."
          
    H = G.copy()
    query_set_size = len(query_nodes)
    map_query_to_org = dict(zip(query_nodes, range(query_set_size)))
    P = csc_matrix(nx.google_matrix(H, alpha=1))
    P_abs = P[list(query_nodes),:][:,list(query_nodes)]
    F = compute_fundamental(P_abs)
    row_sums = start_dist.dot(F.sum())[0,0]
    candidates = list(product(query_nodes, target_nodes))
    eligible = [candidates[i] for i in range(len(candidates)) 
                if H.has_edge(candidates[i][0], candidates[i][1]) == False]
    links_to_add = []
    if alg == 'ra':
        preds = nx.resource_allocation_index(H, eligible)
    elif alg == 'jaccard':
        preds = nx.jaccard_coefficient(H, eligible)
    elif alg == 'aa':
        preds = nx.adamic_adar_index(H, eligible)
    elif alg == 'pa':
        preds = nx.preferential_attachment(H, eligible)
        
    for u,v,p in preds:
        links_to_add.append((u,v,p))
    links_to_add.sort(key=lambda x: x[2], reverse = True)
    
    ac_scores = []
    ac_scores.append(row_sums)
    i = 0
    while i < n_edges:
        F_updated = update_fundamental_mat(F, H, map_query_to_org, links_to_add[i][0])
        H.add_edge(links_to_add[i][0], links_to_add[i][1])
        abs_cen = start_dist.dot(F_updated.sum(axis = 1))[0,0]
        F = F_updated            
        ac_scores.append(abs_cen)
        i += 1
    return links_to_add, ac_scores

def reverse_greedy(G, query_nodes, target_nodes, n_edges, start_dist):
    """Selects a set of links with a reverse greedy descent algorithm that reduce the 
    absorbing RW centrality between a set of query nodes Q and a set of absorbing
    target nodes C such that Q \cap C = \emptyset. The query and target set 
    must be a 'viable' partition of the graph.
    Parameters
    ----------
    G : Networkx graph
        The graph from which the team will be selected.
    query : list 
        The set of nodes from which random walker starts.
    target : list
        The set of nodes from where the random walker ends.
    n_edges : integer
        the number of links to be added
    start_dist: list
        The starting distribution over the query set
    P : Scipy matrix
        The transition matrix of the graph G
    F : Scipy matrix
        The fundamental matrix for the graph G with the given set of absorbing
        random walk nodes
    Returns
    -------
    links : list
        The set of links that reduce the absorbing RW centrality
    """
    H = G.copy()
    query_set_size = len(query_nodes)
    map_query_to_org = dict(zip(query_nodes, range(query_set_size)))
    candidates = list(product(query_nodes, target_nodes))
    eligible = [candidates[i] for i in range(len(candidates)) 
                if H.has_edge(candidates[i][0], candidates[i][1]) == False]
    H.add_edges_from(eligible)
    P = csc_matrix(nx.google_matrix(H, alpha=1))
    P_abs = P[list(query_nodes),:][:,list(query_nodes)]
    F = compute_fundamental(P_abs)
    row_sums = start_dist.dot(F.sum(axis=1))[0,0]
    # candidates = list(product(query_nodes, target_nodes))
    worst_F = zeros(F.shape)
    worst_set = []
    optimal_set = []
    ac_scores = []
#     ac_scores.append(row_sums)
    
    while len(eligible) > 0:
        round_min       = -1
        worst_link      = (-1,-1)
        node_prcessed   = -1
        for out_edge in eligible:
            source_node = out_edge[0]
            if(node_prcessed == source_node):
                # skip updating matrix because this updates the F matrix in the same way
                continue
            node_prcessed = source_node
            F_updated = update_rev_fundamental_mat(F, H, map_query_to_org, source_node)
            abs_cen   = start_dist.dot(F_updated.sum(axis = 1))[0,0]
            if abs_cen < round_min or round_min == -1:
                worst_link  = out_edge
                round_min   = abs_cen
                worst_F     = F_updated
        F = worst_F
        H.remove_edge(*worst_link)
        worst_set.append(worst_link) 
        eligible.remove(worst_link)
        if (len(eligible) <= n_edges):           
            ac_scores.append(round_min)
            optimal_set.append(worst_link)
        
    return list(reversed(optimal_set)), list(reversed(ac_scores))


def get_approx_boundary(G, query_nodes, target_nodes, n_edges, start_dist):
    """
    Used to calculate an approximation guarantee for greedy algorithm
    """
    
    H = G.copy() # GET A COPY OF THE GRAPH
    query_set_size = len(query_nodes) 
    target_set_size = len(target_nodes)
    map_query_to_org = dict(zip(query_nodes, range(query_set_size)))
    
    candidates = list(product(query_nodes, target_nodes))
    # ALL minus exitsting in G
    eligible = [candidates[i] for i in range(len(candidates)) 
                if H.has_edge(candidates[i][0], candidates[i][1]) == False]
    
    # CALCULATE MARGINAL GAIN TO EMPTY SET FOR ALL NODES IN STEEPNESS FUNCTION
    P = csc_matrix(nx.google_matrix(H, alpha=1))
    P_abs = P[list(query_nodes),:][:,list(query_nodes)]
    F = compute_fundamental(P_abs)
    row_sums_empty = start_dist.dot(F.sum(axis=1))[0,0] # F(\emptyset)
    # candidates = list(product(query_nodes, target_nodes))
    ac_marginal_empty   = []
    ac_marginal_full    = []
    source_idx_empty = []
    node_processed = -1
    for out_edge in eligible:
        abs_cen = -1
        source_node = out_edge[0]
        if(node_processed == source_node):
            # skip updating matrix because this updates the F matrix in the same way
            continue
        node_processed = source_node           
        F_updated = update_fundamental_mat(F, H, map_query_to_org, source_node)
        abs_cen = start_dist.dot(F_updated.sum(axis = 1))[0,0]
        ac_marginal_empty.append(abs_cen)
        source_idx_empty.append(source_node)
        
    sorted_indexes_empty = [i[0] for i in sorted(enumerate(source_idx_empty), key=lambda x:x[1])]
    ac_marginal_empty = [ac_marginal_empty[i] for i in sorted_indexes_empty]   
    # CALCULATE MARGINAL GAIN FOR FULL SET

    H.add_edges_from(eligible)
    P_all = csc_matrix(nx.google_matrix(H, alpha=1))
    P_abs_all = P_all[list(query_nodes),:][:,list(query_nodes)]
    F_all = compute_fundamental(P_abs_all)
    
    row_sums_all = start_dist.dot(F_all.sum(axis=1))[0,0]
    node_prcessed   = -1
    source_idx = []
    for out_edge in eligible:
        abs_cen = -1
        source_node = out_edge[0]
        if(node_prcessed == source_node):
            # skip updating matrix because this updates the F matrix in the same way
            continue
        node_prcessed = source_node
        F_all_updated = update_rev_fundamental_mat(F_all, H, map_query_to_org, source_node)
        abs_cen   = start_dist.dot(F_all_updated.sum(axis = 1))[0,0]
        ac_marginal_full.append(abs_cen)
        source_idx.append(source_node)   
    
    sorted_indexes = [i[0] for i in sorted(enumerate(source_idx), key=lambda x:x[1])]
    ac_marginal_full = [ac_marginal_full[i] for i in sorted_indexes]
    
    assert sorted_indexes == sorted_indexes_empty , "Something is wrong with the way scores are appended"
    
    all_steepness = (asarray(ac_marginal_full) - row_sums_all) / (row_sums_empty-asarray(ac_marginal_empty))
    s = min(all_steepness)
    node_max = argmin(all_steepness)
    return 1-s, sorted_indexes[node_max]
     
    
    
     


    

