import numpy as np
import scipy
from scipy import linalg
from scipy import sparse
from scipy import ndimage
import scipy.sparse.linalg.eigen.arpack
from scipy.sparse.linalg.eigen.arpack import eigen, eigen_symmetric
#from pyamg.graph import lloyd_cluster
import pyamg
from pyamg import smoothed_aggregation_solver
from scipy.sparse.linalg import lobpcg
from diffusions import _build_laplacian

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

def spectral_embedding_sparse(adjacency, k_max=14, mode='amg', take_first=True):
    """ A diffusion reordering, but that works for negative values.
    """
    # Normalize the graph: the sum of each set of edges must be one
    diag_weights = np.array(adjacency.sum(axis=1))
    diag_mask = (diag_weights == 0)
    diag_weights[diag_mask] = 1
    dd = np.sign(diag_weights)/np.sqrt(np.abs(diag_weights))
    if mode == 'bf':
        lambdas, diffusion_map = eigen_symmetric(adjacency, k=k_max, which='LA')
        print lambdas
        if take_first:
            res = diffusion_map.T[::-1]*dd.ravel()
        else: 
            res = diffusion_map.T[-2::-1]*dd.ravel()
    elif mode == 'amg':
        print 'amg'
        sh = adjacency.shape[0]
        adjacency = adjacency.copy()
        #diag = sparse.coo_matrix((diag_weights.ravel(), (range(sh), range(sh))))
        diag = sparse.eye(sh, sh)
        adjacency =  - adjacency + diag
        ml = smoothed_aggregation_solver(adjacency.tocsr())
        X = scipy.rand(adjacency.shape[0], k_max) 
        #X[:, 0] = 1. / np.sqrt(adjacency.shape[0])
        X[:, 0] = 1. / dd.ravel()
        M = ml.aspreconditioner()
        lambdas, diffusion_map = lobpcg(adjacency, X, M=M, tol=1.e-12, largest=False)
        print lambdas
        if take_first:
            res = diffusion_map.T * dd.ravel()
        else:
            res = diffusion_map.T[1:] * dd.ravel()
    print res.shape, dd.shape
    return res




def modularity_embedding(adjacency, kmax=10):
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
    indices = np.argsort(lambdas)[::-1]
    print lambdas[:10]
    return maps.T[indices]

def modularity_embedding_sparse(adjacency, kmax=10):
    """ Proceedings of the fifth SIAM international conference on data
        mining, Smyth, A spectral clustering approach to finding
        communities in graphs.

        Return the eigenvalues of the Q matrice
    """
    if isinstance(adjacency, sparse.csc.csc_matrix):
        adjacency = np.array(adjacency.todense())
    abs_adjacency = np.abs(adjacency)
    weights = abs_adjacency/abs_adjacency.sum(axis=0)
    weights = sparse.csc_matrix(weights)
    lambdas, maps = eigen(weights, \
                        k=kmax, which='LR')
    print lambdas
    return maps.T#[1:]




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
    """
    if isinstance(adjacency, sparse.csc.csc_matrix):
        adjacency = np.array(adjacency.todense())
    """
    weights = adjacency
    total_weights = 0.5 * weights.sum()
    for label in np.unique(labels):
        inds = np.nonzero(labels == label)[0]
        a = 0.5 * (weights[inds][:, inds]).sum()
        b = weights[inds].sum() - a
        q += a/total_weights 
        q -= 0.5*(b/total_weights)
        #q += weights[label == labels].T[label == labels].sum()/total_weights 
        #q -= (weights[label == labels].sum()/total_weights)**2
    return 2 * q

def n_cut(adjacency, labels):
    """ Returns the Q score of a clustering.
    """
    q = 0
    """
    if isinstance(adjacency, sparse.csc.csc_matrix):
        adjacency = np.array(adjacency.todense())
    """
    weights = adjacency
    total_weights = 0.5 * weights.sum()
    for label in np.unique(labels):
        inds = np.nonzero(labels == label)[0]
        a = (weights[inds][:, inds]).sum()
        b = weights[inds].sum()
        q += (b - a)/b
    return - q


def best_k_means(k, maps, adjacency, n_bst=10):
    from nipy.neurospin.clustering.clustering import _kmeans
    best_score = -np.inf 
    for _ in range(n_bst):
        print "doing kmeans"
        _, labels, _ = _kmeans(maps, nbclusters=k)
        score2 = q_score(adjacency, labels)
        score = n_cut(adjacency, labels)
        if score > best_score:
            best_score = score
            best_score2 = score2
            best_labels = labels
    return best_labels, best_score2 #best_score


def communities_clustering(adjacency, k_best=None, n_bst=2):
    adjacency = np.abs(adjacency)
    n_features = adjacency.shape[0]
    adjacency.flat[::n_features+1] = 0
    maps = modularity_embedding(adjacency)
    scores = dict()
    if k_best is None:
        #for k in range(2, .3*n_features):
        for k in range(2, 6):
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
    
def communities_clustering_sparse(adjacency, k_best=None, k_min=2, k_max=8, n_bst=4, mode='bf', take_first=False):
    maps = spectral_embedding_sparse(adjacency, k_max=k_max+2, mode=mode, \
                take_first=take_first)
    scores = dict()
    res = dict()
    if k_best is None:
        for k in range(k_min, k_max + 1):
            this_maps = maps[:k - 1].T.copy()
            labels, score = best_k_means(k, this_maps, adjacency, n_bst=n_bst)
            scores[k] = score
            print scores[k]
            res[k] = labels
        #k_best = scores.keys()[np.argmax(scores.values())]
    else:
        this_maps = maps[:k_best - 1].T.copy()
        res, scores = best_k_means(k_best, this_maps, adjacency,
                                 n_bst=4*n_bst)
        print 'Final : k=%i, score=%s' % (k_best, scores)
    return res, scores

def separate_in_regions(data, mask=None, k_best=None, k_min=2, k_max=8, \
                                center=None, only_connex=True, n_times=4,\
                                take_first=True, beta=10, mode='bf'):
    """
    Separate an image in different regions, using spectral clustering.

    Parameters
    ----------

    data: array
        Image to be segmented in regions. `data` can be two- or
        three-dimensional.

    mask: array, optional
        Mask of the pixels to be clustered. If mask is None, all pixels
        are clustered.

    k_best: int, optional
        number of clusters to be found. If k_best is None, the clustering
        is performed for a range of numbers given by k_min and k_max.

    k_min: int, optional
        minimum number of clusters 

    k_max: int, optional
        maximum number of clusters

    center: tuple-like, optional
        coordinates of a point included in the connected component to be
        segmented, if there are several connected components.

    only_connex: boolean, optional
        whether to return only the segmentation of the principal connected 
        component or the (non-clustered) other components as well.

    n_times: int, optional
        how many times the k_means clustering is performed for each k

    take_first: boolean, optional
        whether to take the first eigenmode (of eigenvalue 0) for the clustering
        or not. One should not take it for k=2, but I get better results with it
        for k >= 4 in my images.

    beta: float, optional
        normalization parameter used to compute the weight of a link. The greater
        beta, the more gradients are penalized.

    mode: str, {'bf', 'amg'}
        how the eigenmode of the spectral embedding are computed. 'bf' uses 
        arpack, and 'amg' pyamg (multigrid methods). 'amg' should be much faster.

    Returns
    -------

    labels: array or dict with array values
        result of clustering. If k_best is None, a dict is return and 
        label[k] is the clustering in k clusters, for k_min <= k <= k_max

    scores: int or dict with int values
    """
    if mask is not None:
        labs, nb_labels = ndimage.label(mask)
    else:
        mask = np.ones_like(data).astype(bool)
        nb_labels = 1
    mask = np.atleast_3d(mask)
    if nb_labels > 1:
        if center is None:
            sizes = np.array(ndimage.sum(mask, labs, range(1, nb_labels + 1)))
            ind_max = np.argmax(sizes) + 1
        else:
            ind_max = labs[tuple(center)]
        mask = labs == ind_max
    lap, w = _build_laplacian(np.atleast_3d(data), mask=mask, \
                normed=True, beta=beta)
    print lap.shape
    res, scores = communities_clustering_sparse(lap, k_best=k_best, \
                    k_min=k_min, k_max=k_max, n_bst=n_times, \
                    take_first=take_first, mode=mode)
    mask = np.squeeze(mask)
    if not only_connex:
        if k_best==None:
            labels = dict()
            for k in range(k_min, k_max + 1):
                _ = np.copy(labs)
                _[_ > 0] += k_best
                _[mask > 0] = res[k] + 1
                labels[k] = _    
        else:
            labels = np.copy(labs)
            labels[labels > 0] += k_best
            labels[mask > 0] = res + 1
    else: 
        if k_best==None:
            labels = dict()
            for k in range(k_min, k_max + 1):
                _ = np.copy(mask).astype(np.int)
                _[mask > 0] = res[k] + 1
                labels[k] = _        
        else:
            labels = np.copy(mask).astype(np.int)
            labels[mask > 0] = res + 1
    return labels, scores
