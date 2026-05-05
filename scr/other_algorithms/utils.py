
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
            if B[j, i] == 0:
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

        if i not in intervened_nodes and len(predictors) > 0:
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
    print('Lambda_list:', Lambda_list)
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

