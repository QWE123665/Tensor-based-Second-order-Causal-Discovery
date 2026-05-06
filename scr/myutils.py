
import numpy as np
from itertools import permutations
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LassoLarsIC
from sklearn.preprocessing import StandardScaler
from scipy.linalg import qr as scipy_qr


def covariance(X):
    # X is n by p data matrix
    return np.cov(X.T)

def concentration_list(X_list, B):
    M_list = []
    # Generate covariances by taking lsem samples
    for i in range(B.shape[1]):
        zero_mask = (B[:, i] == 0) # the ith column of binary codes: if =1, hard intervention and zero out this node. 

        Cov_i = covariance(X_list[i])
        M = np.linalg.pinv(Cov_i)
        for j in range(B.shape[0]): 
            if B[j, i] == 0 and Cov_i[j,j]!=0:
                M[j,j]-= 1/Cov_i[j,j]
        M_list.append(M)   
    return M_list


def orthonormal_from_householder_qr(T):
    n,_,k = T.shape
    M = T.reshape(n,n*k)
    Q,R = scipy_qr(M.T, mode='economic')
    T_Q = Q.T.reshape(n,n,k)
    return T_Q



def upper_triangular_mass(A, perm=None):
    """
    Sum of absolute values in the upper triangle of P A P^T.
    k=1 excludes the diagonal, k=0 includes it.
    """
    A = np.asarray(A)
    if perm is not None:
        A = A[np.ix_(perm, perm)]
    mask = np.triu(np.ones(A.shape, dtype=bool), k=1)
    return np.sum(A[mask]**2)



def change_basis_B_svd(T,B):

    n,k = B.shape

    U,D,Vt = np.linalg.svd(B,full_matrices=False)
    if n>=k:
        B_sub = Vt.T@ np.diag(D**-1)

        T_change_basis = (T.reshape(-1,k)@ B_sub).reshape(n,n,k)
    else:
        B_sub = Vt.T@ np.diag(D**-1)

        T_change_basis = (T.reshape(-1,k)@ B_sub).reshape(n,n,n)
    return T_change_basis, U


def random_local_search_pairwise_swaps(
    A,
    current_perm,
    n_restarts=1,
    max_steps=2000,
    random_state=None,
):
    """
    Random local search over pairwise swaps of a matrix permutation to reduce
    upper-triangular mass in P A P^T.

    Returns
    -------
    best_perm : ndarray of shape (n,)
        Permutation with minimized objective found.
    best_score : float
        Upper-triangular mass of A[np.ix_(best_perm, best_perm)].
    best_A : ndarray of shape (n, n)
        Permuted matrix.
    """
    rng = np.random.default_rng(random_state)
    A = np.asarray(A)
    n = A.shape[0]

    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be a square matrix.")

    def score(perm):
        return upper_triangular_mass(A, perm=perm)

    best_perm = current_perm.copy()
    best_score = score(best_perm)

    for _ in range(n_restarts):
        perm = current_perm.copy()
        curr_score = score(perm)

        improved = True
        steps = 0

        while improved and steps < max_steps:
            improved = False
            steps += 1

            pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
            rng.shuffle(pairs)

            for i, j in pairs:
                new_perm = perm.copy()
                new_perm[i], new_perm[j] = new_perm[j], new_perm[i]
                new_score = score(new_perm)

                if new_score < curr_score:
                    perm = new_perm
                    curr_score = new_score
                    improved = True
                    break

        if curr_score < best_score:
            best_perm = perm.copy()
            best_score = curr_score

    best_A = A[np.ix_(best_perm, best_perm)]
    return best_perm, best_score, best_A


### from lingam package

def predict_adaptive_lasso(X, predictors, target, gamma=1.0):
    """Predict with Adaptive Lasso.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Training data, where n_samples is the number of samples
        and n_features is the number of features.
    predictors : array-like, shape (n_predictors)
        Indices of predictor variable.
    target : int
        Index of target variable.

    Returns
    -------
    coef : array-like, shape (n_features)
        Coefficients of predictor variable.
    """
    # Standardize X
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)

    # Pruning with Adaptive Lasso
    lr = LinearRegression()
    lr.fit(X_std[:, predictors], X_std[:, target])
    weight = np.power(np.abs(lr.coef_), gamma)
    reg = LassoLarsIC(criterion="bic")
    reg.fit(X_std[:, predictors] * weight, X_std[:, target])
    pruned_idx = np.abs(reg.coef_ * weight) > 0.0

    # Calculate coefficients of the original scale
    coef = np.zeros(reg.coef_.shape)
    if pruned_idx.sum() > 0:
        lr = LinearRegression()
        pred = np.array(predictors)
        lr.fit(X[:, pred[pruned_idx]], X[:, target])
        coef[pruned_idx] = lr.coef_

    return coef



### from lingam package
def _estimate_adjacency_matrix(causal_order, X, prior_knowledge=None):
    """Estimate adjacency matrix by causal order.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Training data, where n_samples is the number of samples
        and n_features is the number of features.
    prior_knowledge : array-like, shape (n_variables, n_variables), optional (default=None)
        Prior knowledge matrix.

    Returns
    -------
    self : object
        Returns the instance itself.
    """
    if prior_knowledge is not None:
        pk = prior_knowledge.copy()
        np.fill_diagonal(pk, 0)

    A = np.zeros([X.shape[1], X.shape[1]], dtype="float64")
    for i in range(1, len(causal_order)):
        target = causal_order[i]
        predictors = causal_order[:i]

        # Exclude variables specified in no_path with prior knowledge
        if prior_knowledge is not None:
            predictors = [p for p in predictors if pk[target, p] != 0]

        # target is exogenous variables if predictors are empty
        if len(predictors) == 0:
            continue

        A[target, predictors] = predict_adaptive_lasso(X, predictors, target)

    return A


## use interventional data to estimate part of the adjacency matrix.

def _estimate_adjacency_matrix_intervene(causal_order, X, intervened_nodes = None, prior_knowledge=None):
    """Estimate adjacency matrix by causal order.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Training data, where n_samples is the number of samples
        and n_features is the number of features.
    intervened_nodes : array-like, shape (n_intervened_nodes)
        Prior knowledge matrix.

    Returns
    -------
    self : object
        Returns the instance itself.
    """

    if prior_knowledge is not None:
        pk = prior_knowledge.copy()
        np.fill_diagonal(pk, 0)

    A = np.zeros([X.shape[1], X.shape[1]], dtype="float64")
    for i in range(1, len(causal_order)):
        target = causal_order[i]
        predictors = causal_order[:i]

        # Exclude variables specified in no_path with prior knowledge
        if prior_knowledge is not None:
            predictors = [p for p in predictors if pk[target, p] != 0]

        if target not in intervened_nodes and len(predictors) > 0:
            A[target, predictors] = predict_adaptive_lasso(X, predictors, target)

    return A



def improve_Lambda_causal_order_regression_all_pure(X_list, B, node_permutation):
    n,k = B.shape
    Lambda_list = []
    Lambda_obs = _estimate_adjacency_matrix(node_permutation, X_list[0])
    Lambda_list.append(Lambda_obs)
    for i in range(1,len(X_list)):
        Lambda_intervene = _estimate_adjacency_matrix_intervene(
                                node_permutation, 
                                X_list[i], 
                                intervened_nodes = np.where(B[:,i] == 0)[0], prior_knowledge=Lambda_obs
                                )
        Lambda_list.append(Lambda_intervene)
    # print('Lambda_list:', Lambda_list)
    Lambda_2= np.zeros((n,n))
    for i in range(n):
        valid_context = [j for j in range(B.shape[1]) if B[i,j] == 1 and np.sum(np.abs(Lambda_list[j][i,:])) > 0]  
        # contexts where node i is not intervened
        if len(valid_context) > 0:
            Lambda_2[i,:] = np.mean([Lambda_list[j][i,:] for j in valid_context], axis=0)
    return Lambda_2




def relFrob_error(Lambda_est, Lambda_true):
    return np.linalg.norm(Lambda_est - Lambda_true, 'fro') / np.linalg.norm(Lambda_true, 'fro')


def graph_recovery(Lambda_est, Lambda_true):
    est_support = (np.abs(Lambda_est) > 0).astype(int)
    true_support = (np.abs(Lambda_true) > 0).astype(int)
    true_positives = np.sum((est_support == 1) & (true_support == 1))
    false_positives = np.sum((est_support == 1) & (true_support == 0))
    false_negatives = np.sum((est_support == 0) & (true_support == 1))
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    f1_score = 2 * precision * recall / (precision + recall)
    return precision, recall, f1_score

def Ascore(Lambda_est, Lambda_true):
    n = Lambda_true.shape[0]
    M1 = np.eye(n)-Lambda_est.T
    M2 = np.eye(n)-Lambda_true.T
    M1 = M1/np.linalg.norm(M1, axis = 0)
    M2 = M2/np.linalg.norm(M2, axis = 0)
    cos_list = np.sum(M1*M2, axis = 0)
    return np.mean(cos_list)



def _update_cov_list(cov_list, root_idx, root_not_intervened_mask=None):
    """Update covariance matrices by removing root_idx using Schur-complement style adjustment.

    If ``root_not_intervened_mask`` is provided, contexts where root is not intervened
    share an averaged root variance in the denominator.
    """
    updated_cov_list = []
    if root_not_intervened_mask is not None:
        root_not_intervened_mask = np.asarray(root_not_intervened_mask).astype(bool)
        if root_not_intervened_mask.shape[0] != len(cov_list):
            raise ValueError("root_not_intervened_mask must have one entry per context")
        non_intervened_varx = [cov[root_idx, root_idx] for c, cov in enumerate(cov_list) if root_not_intervened_mask[c]]
        avg_varx = float(np.mean(non_intervened_varx)) if len(non_intervened_varx) > 0 else None
    else:
        avg_varx = None

    for c, cov in enumerate(cov_list):

        varx = avg_varx if (avg_varx is not None and root_not_intervened_mask[c]) else cov[root_idx, root_idx]
        v = cov[root_idx, :].reshape(-1, 1)
        if varx!=0:
            updated_cov = cov - v @ v.T / varx
        else:
            updated_cov = cov 
        mask = np.delete(np.arange(cov.shape[0]), root_idx)
        updated_cov = updated_cov[np.ix_(mask, mask)]
        updated_cov_list.append(updated_cov)
    return updated_cov_list, mask




def lambda_from_causal_order_regression(cov, causal_order, threshold=1e-2, ridge=1e-10):
    """Recover Lambda from a covariance matrix via recursive OLS in causal order.

    For each node j in causal order, regresses j on the predecessors
    (nodes earlier in the order) using only the covariance matrix:
        Lambda[j, pred] = Cov(pred, pred)^{-1} @ Cov(pred, j)
    Equivalent to LDL when ``cov`` is PD, but never calls Cholesky — uses
    ``np.linalg.solve`` (LU) per node, which is more forgiving on
    near-singular empirical covariances.

    Parameters
    ----------
    cov : (n, n) array
        Covariance matrix in original variable indexing.
    causal_order : length-n sequence of int
        ``causal_order[0]`` is the root, ``causal_order[-1]`` the sink.
    threshold : float
        Entries of ``Lambda`` with absolute value below this are zeroed.
    ridge : float
        Diagonal jitter on the predecessor block to stabilize the solve.

    Returns
    -------
    Lambda : (n, n) ndarray
        Adjacency matrix in original indexing. ``Lambda[i, j]`` is the
        coefficient of ``X_j`` in the equation for ``X_i``.
    """
    cov = np.asarray(cov, dtype=float)
    perm = np.asarray(causal_order, dtype=int)
    n = cov.shape[0]
    if cov.shape != (n, n):
        raise ValueError("cov must be square")
    if perm.shape != (n,) or set(perm.tolist()) != set(range(n)):
        raise ValueError("causal_order must be a permutation of 0..n-1")

    cov = (cov + cov.T) / 2
    Lambda = np.zeros((n, n))

    for k in range(1, n):
        j = int(perm[k])
        pred = perm[:k]
        Sigma_pp = cov[np.ix_(pred, pred)]
        sigma_pj = cov[pred, j]
        try:
            beta = np.linalg.solve(Sigma_pp + ridge * np.eye(k), sigma_pj)
        except np.linalg.LinAlgError:
            beta = np.linalg.lstsq(Sigma_pp, sigma_pj, rcond=None)[0]
        Lambda[j, pred] = beta

    Lambda[np.abs(Lambda) < threshold] = 0.0
    return Lambda


def cholesky_from_causal_order(cov, causal_order, threshold=1e-2):
    """LDL decomposition of a covariance matrix in a given causal order.

    Permutes ``cov`` so variables appear in causal order (root first), then
    factors ``Sigma_perm = L @ diag(D) @ L.T`` with ``L`` unit lower triangular.
    For an LSEM ``X = Lambda X + eps``, this gives ``L = (I - Lambda_perm)^{-1}``
    and ``D`` are the noise variances in causal order.

    Parameters
    ----------
    cov : (n, n) array
        Covariance matrix in the original variable indexing.
    causal_order : length-n sequence of int
        ``causal_order[0]`` is the root, ``causal_order[-1]`` the sink.

    Returns
    -------
    L : (n, n) ndarray
        Unit lower triangular factor, indexed in causal order.
    D : (n,) ndarray
        Diagonal entries (noise variances) in causal order.
    perm : (n,) ndarray
        ``causal_order`` as an int array.
    Lambda : (n, n) ndarray, optional
        Returned only if ``return_lambda=True``.
    """
    cov = np.asarray(cov, dtype=float)
    perm = np.asarray(causal_order, dtype=int)
    n = cov.shape[0]
    if cov.shape != (n, n):
        raise ValueError("cov must be square")
    if perm.shape != (n,) or set(perm.tolist()) != set(range(n)):
        raise ValueError("causal_order must be a permutation of 0..n-1")

    cov_perm = cov[np.ix_(perm, perm)]
    cov_perm = (cov_perm + cov_perm.T) / 2  # kill antisymmetric numerical noise

    # Jittered Cholesky: empirical covariances often violate PD by tiny
    # amounts. Retry with growing diagonal ridge until it factors.
    scale = float(np.mean(np.diag(cov_perm)))
    scale = scale if scale > 0 else 1.0
    C = None
    last_err = None
    for jitter in (0.0, 1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2):
        try:
            C = np.linalg.cholesky(cov_perm + jitter * scale * np.eye(n))
            if jitter > 0:
                print(f"cholesky_from_causal_order: added jitter {jitter * scale:.2e} "
                      f"to diagonal to factor")
            break
        except np.linalg.LinAlgError as err:
            last_err = err
    if C is None:
        raise np.linalg.LinAlgError(
            f"covariance not positive definite even after jitter; min eig = "
            f"{float(np.linalg.eigvalsh(cov_perm).min()):.3e}"
        ) from last_err

    d_sqrt = np.diag(C)
    D = d_sqrt ** 2
    L = C / d_sqrt                          # unit lower triangular: L[i,j] = C[i,j] / C[j,j]


    Lambda_perm = np.eye(n) - np.linalg.inv(L)
    Lambda = np.zeros_like(Lambda_perm)
    Lambda[np.ix_(perm, perm)] = Lambda_perm
    Lambda[np.abs(Lambda) < threshold] = 0.0
    return Lambda
