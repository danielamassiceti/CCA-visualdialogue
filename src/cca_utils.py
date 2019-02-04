import cv2, torch
import numpy as np
import scipy.linalg as la

def topk_corr_distance(view1, view2, k, dim=0):
  
    denom = torch.norm(view1, p=2, dim=1) * torch.norm(view2.expand_as(view1), p=2, dim=1)
    corr = torch.mm(view1, view2.t()).squeeze().div_(denom)
    return torch.topk(corr, k=k, dim=dim, sorted=True) # indices: top k 

def diagonal(a):
    return a.as_strided((a.size(0),), (a.size(0)+1,))

def mean_center(X, mu):
    return X - mu.t().expand_as(X)

def filter_by_target(x, tgt_idx):
    return [x[0], x[tgt_idx]]

# Bach, F. R. and Jordan, M. I. Kernel independent component analysis. J. Mach. Learn. Res., 3:1-48, 2002
# https://www.di.ens.fr/~fbach/kernelICA-jmlr.pdf
def cca(views, log=None, k=300, eps=1e-12):

    """
    views: list of views, each N x v_i_emb where N is the number of observations and v_i_emb is the embedding dimensionality of that view
    k: integer for the dimensionality of the joint projection space
    eps: float added to diagonals of matrices A and B for stability
    """

    m = views[0].size(0)
    t = views[0].type()
    o = [v.size(1) for v in views]
    os = sum(o)
    A = torch.zeros(os, os).type(t) 
    B = torch.zeros(os, os).type(t)

    if log:
        log.info('doing generalised eigendecomposition...')
   
    row_i = 0
    for i, V_i in enumerate(views):
        V_i = V_i.t()
        o_i = V_i.size(0)
        mu_i = V_i.mean(dim=1, keepdim=True)

        # mean center view i
        V_i_bar = V_i - mu_i.expand_as(V_i) # o_i x N 

        col_i = 0
        for j, V_j in enumerate(views):

            V_j = V_j.t()
            o_j = V_j.size(0) 
            
            if i>j:
                col_i += o_j
                continue
            mu_j = V_j.mean(dim=1, keepdim=True)

            # mean center view j
            V_j_bar = V_j - mu_j.expand_as(V_j) # o_j x N 
            
            C_ij = (1.0 / (m - 1)) * torch.mm(V_i_bar, V_j_bar.t()) # o_i x o_j
            
            A[row_i:row_i+o_i,col_i:col_i+o_j] = C_ij
            A[col_i:col_i+o_j,row_i:row_i+o_i] = C_ij.t()
            if i == j:
                B[row_i:row_i+o_i,col_i:col_i+o_j] = C_ij.clone()
            
            col_i += o_j
        row_i += o_i

    diagonal(A).add_(eps)
    diagonal(B).add_(eps)

    A = A.cpu().numpy()
    B = B.cpu().numpy()

    l, v = la.eig(A, B)
    idx = l.argsort()[-k:][::-1]
    l = l[idx] # eigenvalues
    v = v[:,idx] # eigenvectors

    l = torch.from_numpy(l.real)
    v = torch.from_numpy(v.real)

    # extracting projection matrices
    proj_matrices = [v[sum(o[:i]):sum(o[:i])+views[i].size(1)].type(t) for i in range(len(views))] 
    return l.type(t), proj_matrices

# Mardia, K. V., Kent, J. T., and Bibby, J. M. Multivariate Analysis. Academic Press, 1979.
def cca_mardia(views, log=None, k=300, eps=1e-12, r=1e-4):

    """
    views: list of views, each N x v_i_emb where N is the number of observations and v_i_emb is the embedding dimensionality of that view
    k: integer for the dimensionality of the joint projection space
    eps: float added to diagonals of matrices A and B for stability
    """

    m = views[0].size(0)
    o1 = views[0].size(1)
    o2 = views[1].size(1)
    r1 = r2 = r

    H1 = views[0].t()
    H1bar = H1 - H1.mean(dim=1, keepdim=True).expand_as(H1)
    H2 = views[1].t()
    H2bar = H2 - H2.mean(dim=1, keepdim=True).expand_as(H2)

    SigmaHat12 = (1.0 / (m - 1)) * torch.mm(H1bar, H2bar.t())
    SigmaHat11 = (1.0 / (m - 1)) * torch.mm(H1bar, H1bar.t()) + r1 * torch.eye(o1).type_as(H1)
    SigmaHat22 = (1.0 / (m - 1)) * torch.mm(H2bar, H2bar.t()) + r2 * torch.eye(o2).type_as(H2)

    log.info('doing eigendecomposition...')

    # Calculating the root inverse of covariance matrices by using eigen decomposition
    D1, V1 = torch.symeig(SigmaHat11, eigenvectors=True, upper=False)
    D2, V2 = torch.symeig(SigmaHat22, eigenvectors=True, upper=False)

    # Added to increase stability
    pos_idxs1 = torch.gt(D1, eps).nonzero().squeeze()
    D1 = D1[pos_idxs1]
    V1 = V1[:, pos_idxs1]
    pos_idxs2 = torch.gt(D2, eps).nonzero().squeeze()
    D2 = D2[pos_idxs2]
    V2 = V2[:, pos_idxs2]

    SigmaHat11RootInv = torch.mm(torch.mm(V1, torch.diag(D1 ** -0.5)), V1.t())
    SigmaHat22RootInv = torch.mm(torch.mm(V2, torch.diag(D2 ** -0.5)), V2.t())
    Tval = torch.mm(torch.mm(SigmaHat11RootInv, SigmaHat12), SigmaHat22RootInv)

    U,D,V = torch.svd(Tval)
    D, idx = torch.sort(D, descending=True)
    A1 = torch.mm(SigmaHat11RootInv, U[:,idx][:,:k])
    A2 = torch.mm(SigmaHat22RootInv, V[:,idx][:,:k])
    
    # extracting projection matrices
    proj_matrices = [A1, A2] 
    return D.type(views[0].type()), proj_matrices

def get_projection(x, b, l, p=0):
    return torch.mm(x, torch.mm(b, torch.diag(l ** p)))

