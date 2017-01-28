import cPickle as pickle
import numpy as np
import scipy.sparse

x = np.random.random((10,10))
x = scipy.sparse.csr_matrix(x)

with open('test_sparse_array.dat', 'wb') as outfile:
    pickle.dump(x, outfile, pickle.HIGHEST_PROTOCOL)