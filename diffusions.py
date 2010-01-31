import numpy as np
import scipy, scipy.linalg
from scipy.sparse import coo_matrix, lil_matrix
from pyamg import smoothed_aggregation_solver
from scipy import sparse
import scipy.sparse.linalg.eigen.arpack

#--------- Synthetic data ---------------

def make_2d_syntheticdata(lx, ly=None):
    if ly == None:
        ly = lx
    data = np.zeros((lx, ly)) + 0.1*np.random.randn(lx, ly)
    small_l = int(lx/5)
    data[lx/2-small_l:lx/2+small_l, ly/2-small_l:ly/2+small_l]=1
    data[lx/2-small_l+1:lx/2+small_l-1,\
        ly/2-small_l+1:ly/2+small_l-1]=0.1*np.random.randn(2*small_l-2, 2*small_l-2)
    data[lx/2-small_l, ly/2-small_l/8:ly/2+small_l/8]=0
    seeds = np.zeros_like(data)
    seeds[lx/5, ly/5] = 1
    seeds[lx/2+small_l/4, ly/2-small_l/4] = 2
    return data, seeds

def make_3d_syntheticdata(lx, ly=None, lz=None):
    if ly == None:
        ly = lx
    if lz == None:
        lz = lx
    data = np.zeros((lx, ly, lz)) + 0.1*np.random.randn(lx, ly, lz)
    small_l = int(lx/5)
    data[lx/2-small_l:lx/2+small_l,\
        ly/2-small_l:ly/2+small_l,\
        lz/2-small_l:lz/2+small_l] = 1
    data[lx/2-small_l+1:lx/2+small_l-1,\
        ly/2-small_l+1:ly/2+small_l-1,
        lz/2-small_l+1:lz/2+small_l-1] = 0
    # make a hole
    hole_size = np.max([1, small_l/8])
    data[lx/2-small_l,\
            ly/2-hole_size:ly/2+hole_size,\
            lz/2-hole_size:lz/2+hole_size] = 0
    seeds = np.zeros_like(data)
    seeds[lx/5, ly/5, lz/5] = 1
    seeds[lx/2+small_l/4, ly/2-small_l/4, lz/2-small_l/4] = 2
    return data, seeds


#-----------Laplacian--------------------

def make_edges_3d(lx, ly=None, lz=None):
    if ly == None:
        ly = lx
    if lz == None:
        lz = lx
    vertices = np.arange(lx*ly*lz).reshape((lx, ly, lz))
    edges_deep = np.vstack((vertices[:,:,:-1].ravel(),\
        vertices[:,:,1:].ravel()))
    edges_right = np.vstack((vertices[:,:-1].ravel(), vertices[:,1:].ravel()))
    edges_down = np.vstack((vertices[:-1].ravel(), vertices[1:].ravel()))
    edges = np.hstack((edges_deep, edges_right, edges_down))
    return edges

def make_weights_3d(edges, data, beta=130, eps=1.e-6):
    lx, ly, lz = data.shape
    gradients = (data[edges[0]/(ly*lz),\
                                 (edges[0]%(ly*lz))/lz,\
                                 (edges[0]%(ly*lz))%lz] -\
                            data[edges[1]/(ly*lz),\
                                 (edges[1]%(ly*lz))/lz,\
                                 (edges[1]%(ly*lz))%lz])**2 
    weights = np.exp(-beta*gradients/gradients.max()) + eps
    return weights

def make_laplacian_sparse(edges, weights):
    """
    Sparse implementation
    """
    pixel_nb = len(np.unique(edges.ravel()))
    diag = np.arange(pixel_nb)
    i_indices = np.hstack((edges[0], edges[1]))
    j_indices = np.hstack((edges[1], edges[0]))
    data = np.hstack((-weights, -weights))
    lap = coo_matrix((data, (i_indices, j_indices)), shape=(pixel_nb, pixel_nb))
    connect = -np.ravel(lap.sum(axis=1)) 
    lap = coo_matrix((np.hstack((data, connect)),\
            (np.hstack((i_indices,diag)),\
             np.hstack((j_indices, diag)))), shape=(pixel_nb, pixel_nb))
    return lap.tocsc()

def make_normed_laplacian(edges, weights):
    """
    Sparse implementation
    """
    tol = 1.e-8
    eps = 1.e-5
    pixel_nb = len(np.unique(edges.ravel()))
    diag = np.arange(pixel_nb)
    i_indices = np.hstack((edges[0], edges[1]))
    j_indices = np.hstack((edges[1], edges[0]))
    data = np.hstack((-weights, -weights))
    lap = coo_matrix((data, (i_indices, j_indices)), shape=(pixel_nb, pixel_nb))
    w = -np.ravel(lap.sum(axis=1))
    #w[w<tol] = 1
    data *= 1./(np.sqrt(w[i_indices]*w[j_indices]))
    data = np.hstack((data, eps*np.ones_like(diag)))
    i_indices = np.hstack((i_indices, diag))
    j_indices = np.hstack((j_indices, diag))
    lap = coo_matrix((-data, (i_indices, j_indices)),\
            shape=(pixel_nb, pixel_nb))
    return lap.tocsc(), w

def clean_labels_ar(X, labels):
    labels = np.ravel(labels)
    labels[labels==0] = X
    return labels

def buildAB(lap_sparse, labels):
    lx, ly, lz = labels.shape
    labels = labels.ravel()
    total_ind = np.arange(lx*ly*lz)
    unmarked = total_ind[labels == 0]
    seeds_indices = np.setdiff1d(total_ind, unmarked)
    print "making"
    B = lap_sparse[unmarked][:, seeds_indices]
    lap_sparse = lap_sparse[unmarked][:, unmarked]
    nlabels = labels.max()
    Bi = scipy.sparse.lil_matrix((nlabels,B.shape[0]))
    for lab in range(1, nlabels+1):
        print lab
        fs = scipy.sparse.csr_matrix((labels[seeds_indices] == lab)[:,np.newaxis])
        Bi[lab-1, :] = (B.tocsr()* fs)
    return lap_sparse, Bi
    
def random_walker(data, labels, beta=130, mode='bf'):
    """
        Segmentation with random walker algorithm, given some data and an array of
        labels (the more labels, the less unknowns and the faster the resolution)
        Brute force of multi-grid
    """
    data = np.atleast_3d(data)
    labels = np.atleast_3d(labels)
    print "make lap"
    lap_sparse = build_laplacian(data, beta=beta)
    print "make A, B"
    lap_sparse, B = buildAB(lap_sparse, labels)
    if mode=='bf':
        lap_sparse = lap_sparse.tocsc()
        solver = scipy.sparse.linalg.factorized(lap_sparse.astype(np.double))
        X = np.array([solver(np.array((-B[i,:]).todense()).ravel())\
                for i in range(B.shape[0])])
        X= np.argmax(X, axis=0) + 1
        return (clean_labels_ar(X, labels)).reshape(data.shape)
    elif mode=='amg':
        print "converting"
        lap_sparse = lap_sparse.tocsr()
        print "making mls"
        le = lap_sparse.shape[0]
        print le
        mls = smoothed_aggregation_solver(lap_sparse)
        del lap_sparse
        print "mls ok"
        ll = np.memmap('/tmp/labels', dtype=np.int32, mode='w+', shape=(le,))
        ll[:]=0
        proba_max = np.memmap('/tmp/proba', dtype=np.float32, mode='w+', shape=(le,))
        proba_max[:]=0
        for i in range(B.shape[0]):
            print i
            x = mls.solve(np.ravel(-B[i,:].todense()).astype(np.float32))
            mask = x>proba_max
            ll[mask] = i
            proba_max[mask] = (x[mask]).astype(np.float32)
            del mask
        ll = clean_labels_ar(ll + 1, labels)
        data = np.squeeze(data)
        return ll.reshape(data.shape)



def trim_edges_weights(edges, weights, mask):
    inds = np.arange(mask.size)
    inds = inds[mask.ravel()]
    ind_mask = np.logical_and(np.in1d(edges[0], inds),
                          np.in1d(edges[1], inds))
    edges, weights = edges[:, ind_mask], weights[ind_mask]
    maxval = edges.max()
    order = np.searchsorted(np.unique(edges.ravel()), np.arange(maxval+1))
    edges = order[edges]
    return edges, weights


def build_laplacian(data, mask=None, normed=False, beta=50):
    lx, ly, lz = data.shape
    edges = make_edges_3d(lx, ly, lz)
    weights = make_weights_3d(edges, data, beta=beta, eps=1.e-10)
    if mask is not None:
        edges, weights = trim_edges_weights(edges, weights, mask)
    if not normed:
        lap =  make_laplacian_sparse(edges, weights)
        del edges, weights
        return lap
    else:
        lap, w = make_normed_laplacian(edges, weights)
        del edges, weights
        return lap, w

def fiedler_vector(data, mask, mode='bf'):
    lap, w = build_laplacian(np.atleast_3d(data), 
                np.atleast_3d(mask), normed=True)
    if mode == 'bf':
        vv = scipy.sparse.linalg.eigen.arpack.eigen_symmetric(lap, which='LA', k=5)
        print vv[0]
        values = 1./np.sqrt(w)*vv[1][:,-2]
    if mode == 'amg':
        ml = smoothed_aggregation_solver(lap.tocsr())
        X = scipy.rand(lap.shape[0], 4)
        X[:,0] = 1./np.sqrt(lap.shape[0])
        M = ml.aspreconditioner()
        W,V = scipy.sparse.linalg.lobpcg(-lap, X, M=M, tol=1e-8, largest=True)
        print W
        values = V[:,-2]
    result = np.zeros_like(data).astype(np.float)
    result[mask] = values
    return result




def random_walker_prior(data, prior, mode='bf', gamma=1.e-2):
    k = prior.shape[1]
    data = np.atleast_3d(data)
    print "building lap"
    lap_sparse = build_laplacian(data)
    print "lap ok"
    #lap_sparse = lap_sparse.tolil()
    dia = range(data.size)
    lap_sparse = lap_sparse +scipy.sparse.lil_diags([gamma*prior.sum(axis=0)],[0],\
        lap_sparse.shape)
    del dia
    if mode=='bf':
        lap_sparse = lap_sparse.tocsc()
        solver = scipy.sparse.linalg.factorized(lap_sparse.astype(np.double))
        X = np.array([solver(gamma*label_prior)\
                for label_prior in prior])
    elif mode=='amg':
        print "converting"
        lap_sparse = lap_sparse.tocsr()
        print "making mls"
        mls = smoothed_aggregation_solver(lap_sparse)
        del lap_sparse
        print "mls ok"
        X = np.array([mls.solve(gamma*label_prior)\
                for label_prior in prior])
        del mls
    return np.argmax(X, axis=0)



#----------- Tests --------------------------------

def test_2d():
    lx = 70
    ly = 100
    data, labels = make_2d_syntheticdata(lx, ly)
    labels = random_walker(data, labels, beta=90)
    #assert (labels.reshape((lx, ly))[25:45, 40:60] == 2).all()
    return data, labels

def test_3d():    
    n=30
    lx, ly, lz = n, n, n
    data, labels = make_3d_syntheticdata(lx, ly, lz)
    labels = random_walker(data, labels)
    assert (labels.reshape(data.shape)[13:17,13:17,13:17] == 2).all()
    return data, labels

