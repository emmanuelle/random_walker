"""
diffusions: a module providing segmentation algorithm based on diffusion.

The algorithms of this module are based on the "random walker" algorithm.
"""

# Author: Emmanuelle Gouillart <emmanuelle.gouillart@normalesup.org>
# Copyright (c) 2009-2010, Emmanuelle Gouillart
# License: BSD

import warnings

import numpy as np
from scipy import sparse, ndimage
try: 
    from scipy.sparse.linalg.dsolve import umfpack
    u = umfpack.UmfpackContext()
except:
    warnings.warn("""Scipy was built without UMFPACK. Consider rebuilding 
    Scipy with UMFPACK, this will greatly speed up the random walker 
    functions. You may also install pyamg and run the random walker function 
    in amg mode (see the docstrings)
    """)
try:
    from pyamg import smoothed_aggregation_solver
    amg_loaded = True
except ImportError:
    amg_loaded = False 
import scipy
scipy_version = scipy.__version__.split('.')




#-----------Laplacian--------------------

def _make_edges_3d(lx, ly, lz):
    """ Returns a list of edges for a 3D image.
    
        Parameters
        ===========
        lx: integer
            The size of the grid in the x direction.
        ly: integer, optinal
            The size of the grid in the y direction, defaults
            to lx.
        lz: integer, optinal
            The size of the grid in the z direction, defaults
            to lx.
    """
    vertices = np.arange(lx*ly*lz).reshape((lx, ly, lz))
    edges_deep = np.vstack((vertices[:, :, :-1].ravel(),
                            vertices[:, :, 1:].ravel()))
    edges_right = np.vstack((vertices[:, :-1].ravel(), vertices[:, 1:].ravel()))
    edges_down = np.vstack((vertices[:-1].ravel(), vertices[1:].ravel()))
    edges = np.hstack((edges_deep, edges_right, edges_down))
    return edges

def _make_weights_3d(edges, data, beta=130, eps=1.e-6):
    lx, ly, lz = data.shape
    gradients = _make_distances_3d(edges, data)**2 
    weights = np.exp(- beta*gradients / (10*data.std())) + eps
    return weights

def _make_distances_3d(edges, data):
    lx, ly, lz = data.shape
    gradients = np.abs(data[edges[0]/(ly*lz), \
                                 (edges[0] % (ly*lz))/lz, \
                                 (edges[0] % (ly*lz))%lz] - \
                            data[edges[1]/(ly*lz), \
                                 (edges[1] % (ly*lz))/lz, \
                                 (edges[1] % (ly*lz)) % lz])
    return gradients

def _make_laplacian_sparse(edges, weights):
    """
    Sparse implementation
    """
    pixel_nb = len(np.unique(edges.ravel()))
    diag = np.arange(pixel_nb)
    i_indices = np.hstack((edges[0], edges[1]))
    j_indices = np.hstack((edges[1], edges[0]))
    data = np.hstack((-weights, -weights))
    lap = sparse.coo_matrix((data, (i_indices, j_indices)), 
                            shape=(pixel_nb, pixel_nb))
    connect = - np.ravel(lap.sum(axis=1)) 
    lap = sparse.coo_matrix((np.hstack((data, connect)),
                (np.hstack((i_indices,diag)), np.hstack((j_indices, diag)))), 
                shape=(pixel_nb, pixel_nb))
    return lap.tocsc()

def _clean_labels_ar(X, labels):
    labels = np.ravel(labels)
    labels[labels == 0] = X
    return labels

def _buildAB(lap_sparse, labels):
    lx, ly, lz = labels.shape
    labels = labels.ravel()
    labels = labels[labels >= 0]
    total_ind = np.arange(labels.size) 
    unmarked = total_ind[labels == 0]
    seeds_indices = np.setdiff1d(total_ind, unmarked)
    B = lap_sparse[unmarked][:, seeds_indices]
    lap_sparse = lap_sparse[unmarked][:, unmarked]
    nlabels = labels.max()
    Bi = sparse.lil_matrix((nlabels, B.shape[0]))
    for lab in range(1, nlabels+1):
        fs = sparse.csr_matrix((labels[seeds_indices] == lab)\
                                                [:, np.newaxis])
        Bi[lab-1, :] = (B.tocsr()* fs)
    return lap_sparse, Bi
    
def _trim_edges_weights(edges, weights, mask):
    inds = np.arange(mask.size)
    inds = inds[mask.ravel()]
    ind_mask = np.logical_and(np.in1d(edges[0], inds),
                          np.in1d(edges[1], inds))
    edges, weights = edges[:, ind_mask], weights[ind_mask]
    maxval = edges.max()
    order = np.searchsorted(np.unique(edges.ravel()), np.arange(maxval+1))
    edges = order[edges]
    return edges, weights


def _build_laplacian(data, mask=None, normed=False, beta=50):
    lx, ly, lz = data.shape
    edges = _make_edges_3d(lx, ly, lz)
    weights = _make_weights_3d(edges, data, beta=beta, eps=1.e-10)
    if mask is not None:
        edges, weights = _trim_edges_weights(edges, weights, mask)
    if not normed:
        lap =  _make_laplacian_sparse(edges, weights)
        del edges, weights
        return lap
    else:
        lap, w = _make_normed_laplacian(edges, weights)
        del edges, weights
        return lap, w



#----------- Random walker algorithms (with markers or with prior) -------------

@profile
def random_walker(data, labels, beta=130, mode='bf', copy=True):
    """
        Segmentation with random walker algorithm by Leo Grady, 
        given some data and an array of labels (the more labeled 
        pixels, the less unknowns and the faster the resolution)

        Parameters
        ----------

        data : array_like
            Image to be segmented in regions. `data` can be two- or
            three-dimensional.

        labels : array of ints
            Array of seed markers labeled with different integers
            for different phases. Negative labels correspond to inactive
            pixels that do not diffuse (they are removed from the graph).

        beta : float
            Penalization coefficient for the random walker motion
            (the greater `beta`, the more difficult the diffusion).

        mode : {'bf', 'amg'}
            Mode for solving the linear system in the random walker 
            algorithm. `mode` can be either 'bf' (for brute force),
            in which case matrices are directly inverted, or 'amg'
            (for algebraic multigrid solver), in which case a multigrid
            approach is used. The 'amg' mode uses the pyamg module 
            (http://code.google.com/p/pyamg/), which must be installed
            to use this mode.

        copy : bool
            If copy is False, the `labels` array will be overwritten with
            the result of the segmentation. Use copy=False if you want to 
            save on memory.

        Returns
        -------

        output : ndarray of ints
            Array in which each pixel has been attributed the label number
            that reached the pixel first by diffusion.

        Notes
        -----

        The algorithm was first proposed in *Random walks for image 
        segmentation*, Leo Grady, IEEE Trans Pattern Anal Mach Intell. 
        2006 Nov;28(11):1768-83.

        Examples
        --------

        >>> a = np.zeros((10, 10)) + 0.2*np.random.random((10, 10))
        >>> a[5:8, 5:8] += 1
        >>> b = np.zeros_like(a)
        >>> b[3,3] = 1 #Marker for first phase
        >>> b[6,6] = 2 #Marker for second phase
        >>> random_walker(a, b)
        array([[ 1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.],
               [ 1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.],
               [ 1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.],
               [ 1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.],
               [ 1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.],
               [ 1.,  1.,  1.,  1.,  1.,  2.,  2.,  2.,  1.,  1.],
               [ 1.,  1.,  1.,  1.,  1.,  2.,  2.,  2.,  1.,  1.],
               [ 1.,  1.,  1.,  1.,  1.,  2.,  2.,  2.,  1.,  1.],
               [ 1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.],
               [ 1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.]])

    """
    # We work with 3-D arrays
    data = np.atleast_3d(data)
    if copy:
        labels = np.copy(labels)
    labels = labels.astype(np.int)
    # If the array has pruned zones, be sure that no isolated pixels
    # exist between pruned zones (they could not be determined)
    if np.any(labels<0):
        filled = ndimage.binary_propagation(labels>0, mask=labels>=0)
        labels[np.logical_and(np.logical_not(filled), labels == 0)] = -1
        del filled
    labels = np.atleast_3d(labels)
    if np.any(labels < 0):
        lap_sparse = _build_laplacian(data, mask=labels >= 0, beta=beta)
    else:
        lap_sparse = _build_laplacian(data, beta=beta)
    lap_sparse, B = _buildAB(lap_sparse, labels)
    # We solve the linear system
    # lap_sparse X = B
    # where X[i, j] is the probability that a marker of label i arrives 
    # first at pixel j by diffusion
    if mode == 'bf':
        lap_sparse = lap_sparse.tocsc()
        solver = sparse.linalg.factorized(lap_sparse.astype(np.double))
        X = np.array([solver(np.array((-B[i, :]).todense()).ravel())\
                for i in range(B.shape[0])])
        X = np.argmax(X, axis=0) + 1
        data = np.squeeze(data)
        return (_clean_labels_ar(X, labels)).reshape(data.shape)
    elif mode == 'amg':
        if not amg_loaded:
            print """the pyamg module (http://code.google.com/p/pyamg/)
            must be installed to use the amg mode"""
            raise ImportError
        lap_sparse = lap_sparse.tocsr()
        le = lap_sparse.shape[0]
        mls = smoothed_aggregation_solver(lap_sparse)
        del lap_sparse
        ll = np.zeros(le, dtype=np.int32)
        proba_max = np.zeros(le, dtype=np.float32)
        for i in range(B.shape[0]):
            x = mls.solve(np.ravel(-B[i, :].todense()).astype(np.float32))
            mask = x > proba_max
            ll[mask] = i
            proba_max[mask] = (x[mask]).astype(np.float32)
            del mask
        del proba_max
        ll = _clean_labels_ar(ll + 1, labels)
        data = np.squeeze(data)
        return ll.reshape(data.shape)


def random_walker_prior(data, prior, mode='bf', gamma=1.e-2):
    """
        Parameters
        ----------

        data : array_like
            Image to be segmented in regions. `data` can be two- or
            three-dimensional.


        prior : array_like
            Array of 1-dimensional probabilities that the pixels 
            belong to the different phases. The size of `prior` is n x s
            where n is the number of phases to be segmented, and s the 
            total number of pixels.

        mode : {'bf', 'amg'}
            Mode for solving the linear system in the random walker 
            algorithm. `mode` can be either 'bf' (for brute force),
            in which case matrices are directly inverted, or 'amg'
            (for algebraic multigrid solver), in which case a multigrid
            approach is used. The 'amg' mode uses the pyamg module 
            (http://code.google.com/p/pyamg/), which must be installed
            to use this mode.

        gamma : float
            gamma determines the absence of confidence into the prior. 
            The smaller gamma, the more the output values will be determined
            according to the prior only. Conversely, the greater gamma, 
            the more continuous the segmented regions will be.

        Returns
        -------

        output : ndarray of ints
            Segmentation of data. The number of phases corresponds to the
            number of lines in prior.

        Notes
        -----

        The algorithm was first proposed in *Multilabel random walker 
        image segmentation using prior models*, L. Grady, IEEE CVPR 2005,
        p. 770 (2005).

        Examples
        --------
        >>> a = np.zeros((40, 40))
        >>> a[10:-10, 10:-10] = 1
        >>> a += 0.7*np.random.random((40, 40))
        >>> p = a.max() - a.ravel()
        >>> q = a.ravel()
        >>> prior = np.array([p, q])
        >>> labs = random_walker_prior(a, prior)
    """
    data = np.atleast_3d(data)
    lap_sparse = _build_laplacian(data, beta=50)
    dia = range(data.size)
    shx, shy = lap_sparse.shape
    lap_sparse = lap_sparse + sparse.coo_matrix(
                        (gamma*prior.sum(axis=0), (range(shx), range(shy))))
    del dia
    if mode == 'bf':
        lap_sparse = lap_sparse.tocsc()
        solver = sparse.linalg.factorized(lap_sparse.astype(np.double))
        X = np.array([solver(gamma*label_prior)
                      for label_prior in prior])
    elif mode == 'amg':
        if not amg_loaded:
            print """the pyamg module (http://code.google.com/p/pyamg/)
            must be installed to use the amg mode"""
            raise ImportError
        lap_sparse = lap_sparse.tocsr()
        mls = smoothed_aggregation_solver(lap_sparse)
        del lap_sparse
        X = np.array([mls.solve(gamma*label_prior)
                      for label_prior in prior])
        del mls
    return np.squeeze((np.argmax(X, axis=0)).reshape(data.shape))





 
