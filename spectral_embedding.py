
import numpy as np
from scipy import linalg

def spectral_embedding(adjacency):
    """ A diffusion reordering, but that works for negative values.
    """
    # Normalize the graph: the sum of each set of edges must be one
    abs_adjacency = np.abs(adjacency)
    diag_weights = abs_adjacency.sum(axis=1)
    diag_mask = (diag_weights == 0)
    diag_weights[diag_mask] = 1
    d = np.sign(diag_weights)/np.sqrt(np.abs(diag_weights))
    lap = abs_adjacency*d[:, np.newaxis]*d[np.newaxis, :]
    lambdas, diffusion_map = linalg.eigh(lap)
    return lambdas, diffusion_map.T[-2::-1]*d



def modularity_embedding(adjacency):
    """ Proceedings of the fifth SIAM international conference on data
        mining, Smyth, A spectral clustering approach to finding
        communities in graphs.

        Return the eigenvalues of the Q matrice
    """
    #n = len(adjacency)
    abs_adjacency = np.abs(adjacency)
    #degrees = adjacency.copy()
    #degrees.flat[::n+1] = 0
    #degrees = degrees.sum(axis=0)
    #weights = 1/degrees[:, np.newaxis] * abs_adjacency 
    #weights.flat[::n+1] = 1
    weights = abs_adjacency/abs_adjacency.sum(axis=0)
    lambdas, maps = linalg.eig(weights)
    return maps.T


def newman_clustering(adjacency, eps=1e-8):
    """ Newmann's spectral embedding algorithm to maximize modularity.
    """
    n = len(adjacency)
    abs_adjacency = np.abs(adjacency)
    abs_adjacency.flat[::n+1] = 0
    degrees = abs_adjacency.sum(axis=0)
    weights = abs_adjacency - np.dot(degrees[:, np.newaxis],
                                     degrees[np.newaxis, :])/degrees.sum()
    weights.flat[::n+1] = 0
    weights -= np.diag(weights.sum(axis=0))
    lambdas, maps = linalg.eigh(weights)
    if lambdas[-1] <= eps:
        return np.ones(n, dtype=np.int)
    cluster1 = maps.T[-1] >= 0
    cluster2 = maps.T[-1] <  0
    labels = np.zeros(n, dtype=np.int)
    labels[cluster1] = 2*newman_clustering(adjacency[cluster1].T[cluster1])
    labels[cluster2] = (1+
                    2*newman_clustering(adjacency[cluster2].T[cluster2])
                    )
    return labels


def q_score(adjacency, labels):
    """ Returns the Q score of a clustering.
    """
    q = 0
    n_features = adjacency.shape[0]
    weights = np.abs(adjacency)
    weights.flat[::n_features+1] = 0
    total_weights = weights.sum()
    for label in np.unique(labels):
        q += weights[label == labels].T[label == labels].sum()/total_weights 
        q -= (weights[label == labels].sum()/total_weights)**2
    return q


def best_k_means(k, maps, adjacency, n_bst=10):
    from nipy.neurospin.clustering.clustering import _kmeans
    best_score = -np.inf 
    for _ in range(n_bst):
        _, labels, _ = _kmeans(maps, nbclusters=k)
        score = q_score(adjacency, labels)
        if score > best_score:
            best_score = score
            best_labels = labels
    return best_labels, best_score


def communities_clustering(adjacency, k_best=None, n_bst=10):
    adjacency = np.abs(adjacency)
    n_features = adjacency.shape[0]
    adjacency.flat[::n_features+1] = 0
    maps = modularity_embedding(adjacency)
    scores = dict()
    if k_best is None:
        for k in range(2, .3*n_features):
            this_maps = maps[:k-1].T.copy()
            labels, score = best_k_means(k, this_maps, adjacency, n_bst=n_bst)
            scores[k] = score
            print scores[k]
        k_best = scores.keys()[np.argmax(scores.values())]
    this_maps = maps[:k_best-1].T.copy()
    labels, score = best_k_means(k_best, this_maps, adjacency,
                                 n_bst=5*n_bst)
    print 'Final : k=%i, score=%s' % (k_best, score)
    return labels
