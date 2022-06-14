import numpy as np
from scipy.sparse import coo_matrix
from sklearn.neighbors import NearestNeighbors
import scipy.io as io
import scipy as sp

def make_sym(K):
    maxit = 100
    N = K.shape[0]
    r = np.ones(N)
    for i in range(maxit):
        c = 1./ (np.transpose(K) * r)
        r = 1./ (K * c)                 #  RuntimeWarning: divide by zero encountered in true_divide
    C = sp.sparse.spdiags(c, 0, N, N)
    R = sp.sparse.spdiags(r, 0, N, N)

    newK = R * K * C
    newK = (np.transpose(newK) + newK) / 2
    return newK

def buildSparseK(N, W, J, numvox, flag):
    # --- create K ------------------------------------------------------
    row = J
    col = J[N[:, 0]]
    K = coo_matrix((W[:, 0], (row, col)), shape=(numvox, numvox))
    for n in range(2, N.shape[1]):
        row_n = J
        col_n = J[N[:, n]]
        Kn = coo_matrix((W[:, n], (row_n, col_n)), shape=(numvox, numvox))
        K = K + Kn
    # --- prevent any all-zeros rows -------------------------------------
    sumK = np.sum(K, axis=1)                      # sum之后 直接转为完全矩阵， 不需要full, 不同于matlab ???
    I = np.where(sumK == 0)[0]
    for i in range(len(I)):
        K[I[i], I[i]] = 1
    # --- make symmetric -------------------------------------------------
    if flag:
        K_trans = np.transpose(K)
        K = (K_trans + K) / 2
        K = make_sym(K)
    return K

def buildKernel(imgsiz, nbrtyp, nbrpar, X, kertyp, kerpar, normflag, midflag, inplaneflag):
    imgdim = len(imgsiz)
    if len(nbrpar) == 1:
        if imgdim ==3:
            nbrpar = [nbrpar[0], nbrpar[0], nbrpar[0]]
        elif imgdim ==2:
            nbrpar = [nbrpar[0], nbrpar[0], 1]
    if nbrtyp == 'knn':
        # --- knn cal --------------------------------------------------
        k = nbrpar[0]
        neigh = NearestNeighbors(n_neighbors=k)
        neigh.fit(X)
        knn_output = neigh.kneighbors(X)
        D = knn_output[0]
        N = knn_output[1]
        # --- cal w ----------------------------------------------------
        W = calc_wgt(X, N, kertyp, kerpar)
        # --- normalized to 1 and output -------------------------------
        if normflag:
            sumw = np.zeros(W.shape)
            for i in range(W.shape[1]):
                sumw[:, i] = np.sum(W, axis=1) # 行求和
            W = W / sumw
            W[sumw == 0] = 0
    # if nbrtyp == 'cube':
    #     pass
    return N, W


def calc_wgt(X, N, kertyp, kerpar):
    # --- set J ------------------------------------------
    J = N.shape[0]
    # --- set D ------------------------------------------
    D = np.zeros(N.shape)
    for i in range(N.shape[1]):
        dataX_modify = (X[0:J, :] - X[N[:,i], :])
        #--- sparse --------------------------------------
        row = np.arange(X.shape[1])
        col = np.arange(X.shape[1])
        data_std = 1./np.std(X, axis=0)
        sparse_matrix = coo_matrix((data_std, (row, col)))
        # --- get D ---------------------------------------
        D[:,i] = (np.mean(np.power(dataX_modify * sparse_matrix, 2),1)) ** 0.5
    # --- switch select -----------------------------------
    switch = {'invdist': case1,
              'radial': case2,
              'poly': case3,
              'nnlle': case4,
              }
    W = switch.get(kertyp, default)(D, kerpar)  # 执行对应的函数，如果没有就执行默认的函数
    return W

# --- define switch ----------------------------------------
def case1(D):
    print('pass  kernel invdist ---------------------------')
    W = 1./D
    return W
def case2(D, kerpar):
    print('--- pass  kernel radial -------------------------')
    W = np.exp((np.power(-D, 2))/(np.power(2*kerpar, 2)))
    return W
def case3(N, kerpar, J):
    print('--- pass  kernel poly ---------------------------')
    W = np.zeros(N.shape)
    for i in range(N.shape[1]):
        W[:, i] = np.power((np.mean(X[0:J, :] * X[N[:,i], :],1) + kerpar), 2)
    return W
def case4():
    print('--- Not support kernel nnlle, sorry so much! ---')
def default():                          # 默认情况下执行的函数
    print('unknown kernel type')


if __name__ == "__main__":
    # --- parameter setting ----------------------------------
    # --- parameter of buildKernel ---------------------------
    imgsiz = [128, 128]
    nbrtyp = 'knn'
    nbrpar = [48]
    #X = np.array([[1,2,3,3,2],[4,5,6,6,3],[7,8,9,9,6]])
    matr = io.loadmat(r'U_2s.mat')
    data = matr['U']
    #numpy_data = np.transpose(data)
    print(data.shape)
    print(np.max(data))
    kertyp = 'radial'
    kerpar = 1
    normflag = 1
    midflag = 1
    inplaneflag = 0
    # --- function -------------------------------------------
    N, W = buildKernel(imgsiz, nbrtyp, nbrpar, data, kertyp, kerpar, normflag, midflag, inplaneflag)
    # --- parameter of buildSparseK --------------------------
    J = np.arange(N.shape[0])  # 向量size = m
    numvox = N.shape[0]
    flag = 1

    #matr1 = io.loadmat(r'N.mat')
    #N = matr1['N']
    #matr2 = io.loadmat(r'W.mat')
    #W = matr2['W']
    K = buildSparseK(N, W, J, numvox, flag)
    print(K)
    # #W = W / np.std(data)
    # print(N.shape, '*******************')
    # print(W.shape, '*******************')
    # print(np.std(data))
    # #io.savemat('N.mat', {'N': N})
    io.savemat('K1.mat', {'K1': K})
    # print('done')
