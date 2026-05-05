from __future__ import annotations

import numpy as np



def similarity_measures(B,A):
    J=A.shape[1]
    columnlist=[]
    for i in range(J):
        v=A[:,i]
        dislist=[]
        signlist=[]
        for j in range(B.shape[1]):
            u=abs(v-B[:,j])
            uprime=abs(v+B[:,j])
            sign=1
            if np.sum(u**2)<np.sum(uprime**2):
                dislist.append(np.sum(u))
                signlist.append(sign)
            else:
                sign=-1
                dislist.append(np.sum(uprime**2))
                signlist.append(sign)
        a=min(dislist)
        index=dislist.index(a)
        sign=signlist[index]
        columnlist.append(sign*B[:,index])
        B=np.delete(B,index,1)
    permutedB=np.transpose(np.vstack(tuple(columnlist)))
    cosine_similarity=np.mean(np.sum(permutedB*A,axis=0))
    # for now the bound is 0.95
    return cosine_similarity,np.sort(np.sum(permutedB*A,axis=0))[::-1]

# Metrics
# ===============================

def binarize_adjacency(
    W: np.ndarray,
    threshold: float =0,
    include_diagonal: bool = False
) -> np.ndarray:
    """
    Convert weigh matrix W into a binary adjacency matrix A:
      A[i,j] = 1  iff  |W[i,j]| > threshold

    Args:
        W: (n,n) array-like
        threshold: edge-present threshold
        include_diagonal: if False, diagonal forced to 0

    Returns:
        (n,n) binary int matrix
    """
    W = np.asarray(W)
    if W.ndim != 2 or W.shape[0] != W.shape[1]:
        raise ValueError(f"W must be square (n,n). Got {W.shape}.")

    A = (np.abs(W) > threshold).astype(int)
    if not include_diagonal:
        np.fill_diagonal(A, 0)
    return A


# deal with partial graphs (contain undirected edges)
def tpr_p(A_true: np.ndarray,
    A_hat: np.ndarray
) -> float:    
    """
    True Positive Rate (Recall) for directed edges:
        TPR = TP / (TP + FN)
    Args:
        A_true: binary ground-truth adjacency (n,n)
        A_hat:  binary estimated adjacency (n,n)
    Returns:
        float in [0,1]; returns 1.0 if there are no true edges.
    """
    A_true = np.asarray(A_true).astype(int)
    A_hat = np.asarray(A_hat).astype(int)
    if A_true.shape != A_hat.shape:
        raise ValueError(f"Shape mismatch: {A_true.shape} vs {A_hat.shape}")
    if A_true.ndim != 2 or A_true.shape[0] != A_true.shape[1]:
        raise ValueError("Adjacency matrices must be square (n,n).")

    # Treat an undirected edge as present if either direction is present.
    true_present = (np.abs(A_true) > 0)
    hat_present = (np.abs(A_hat) > 0)
    true_present = true_present | true_present.T
    hat_present = hat_present | hat_present.T
    np.fill_diagonal(true_present, False)
    np.fill_diagonal(hat_present, False)

    tp = np.sum(true_present & hat_present)
    fn = np.sum(true_present & ~hat_present)
    denom = tp + fn
    return 1.0 if denom == 0 else tp / denom


def tpr(
    A_true: np.ndarray,
    A_hat: np.ndarray
) -> float:
    """
    True Positive Rate (Recall) for directed edges:
      TPR = TP / (TP + FN)

    Args:
        A_true: binary ground-truth adjacency (n,n)
        A_hat:  binary estimated adjacency (n,n)

    Returns:
        float in [0,1]; returns 1.0 if there are no true edges.
    """
    A_true = np.asarray(A_true).astype(int)
    A_hat = np.asarray(A_hat).astype(int)
    if A_true.shape != A_hat.shape:
        raise ValueError(f"Shape mismatch: {A_true.shape} vs {A_hat.shape}")

    tp = np.sum((A_true == 1) & (A_hat == 1))
    fn = np.sum((A_true == 1) & (A_hat == 0))
    denom = tp + fn
    return 1.0 if denom == 0 else tp / denom


def shd(
    A_true: np.ndarray,
    A_hat: np.ndarray,
    count_reversals_as_one: bool = True
) -> int:
    """
    Structural Hamming Distance (SHD) between two directed graphs.

    If count_reversals_as_one=True:
      - extra edge (FP): +1
      - missing edge (FN): +1
      - reversed edge i->j vs j->i: +1 total (not +2)

    Args:
        A_true: binary ground-truth adjacency (n,n)
        A_hat:  binary estimated adjacency (n,n)
        count_reversals_as_one: whether reversal counts as 1 edit

    Returns:
        nonnegative int
    """
    A_true = np.asarray(A_true).astype(int)
    A_hat = np.asarray(A_hat).astype(int)
    if A_true.shape != A_hat.shape:
        raise ValueError(f"Shape mismatch: {A_true.shape} vs {A_hat.shape}")
    if A_true.ndim != 2 or A_true.shape[0] != A_true.shape[1]:
        raise ValueError("Adjacency matrices must be square (n,n).")

    if count_reversals_as_one:
        # For each unordered pair {i,j}, count:
        # 0 if both directions match (both 0/1),
        # 1 if it's a reversal or single-edge mismatch,
        # 2 if they disagree on both directions (very rare if both are DAGs, but handled).
        n = A_true.shape[0]
        total = 0
        for i in range(n):
            for j in range(i + 1, n):
                a_ij, a_ji = A_true[i, j], A_true[j, i]
                h_ij, h_ji = A_hat[i, j], A_hat[j, i]

                # number of directed mismatches for this pair
                pair_mismatches = int(a_ij != h_ij) + int(a_ji != h_ji)

                # If exactly one directed mismatch -> one add/delete
                # If two mismatches:
                #   - could be a reversal (1->0 and 0->1): count as 1
                #   - or could be (1,1) vs (0,0) etc: count as 2
                if pair_mismatches == 2 and (a_ij + a_ji == 1) and (h_ij + h_ji == 1):
                    total += 1  # reversal
                else:
                    total += pair_mismatches
        return total
    else:
        # Pure directed Hamming distance: every differing entry counts as 1
        return int(np.sum(A_true != A_hat))


# reversal count as one
# this function is for outputs of causallearn,
# A[i,j] = 1, A[j,i] = -1 means edge i->j
# A[i,j] = A[j,i] = 1 or -1 means undirected edge i-j
def shd_binary(
    A_true: np.ndarray,
    A_hat: np.ndarray
) -> int:
    """
    Structural Hamming Distance (SHD) between two directed graphs.

    If count_reversals_as_one=True:
      - extra edge (FP): +1
      - missing edge (FN): +1
      - reversed edge i->j vs j->i: +1 total (not +2)

    Args:
        A_true: binary ground-truth adjacency (n,n)
        A_hat:  binary estimated adjacency (n,n)
        count_reversals_as_one: whether reversal counts as 1 edit

    Returns:
        nonnegative int
    """
    A_true = np.asarray(A_true).astype(int)
    A_hat = np.asarray(A_hat).astype(int)
    if A_true.shape != A_hat.shape:
        raise ValueError(f"Shape mismatch: {A_true.shape} vs {A_hat.shape}")
    if A_true.ndim != 2 or A_true.shape[0] != A_true.shape[1]:
        raise ValueError("Adjacency matrices must be square (n,n).")

    # For each unordered pair {i,j}, count:
    # 0 if both directions match (both 0/1),
    # 1 if it's a reversal or single-edge mismatch,
    # 2 if they disagree on both directions (very rare if both are DAGs, but handled).
    n = A_true.shape[0]
    total = 0
    for i in range(n):
        for j in range(i + 1, n):
            a_ij, a_ji = A_true[i, j], A_true[j, i]
            h_ij, h_ji = A_hat[i, j], A_hat[j, i]

            # number of directed mismatches for this pair
            edge_mismatches = int(abs(h_ij) != max(np.abs(a_ij),np.abs(a_ji))) 
            if edge_mismatches == 0 and (h_ij + h_ji == 0) and ((h_ij + a_ij == 0) or (h_ji + a_ji == 0)):
                total += 1  # reversal
            else:
                total += edge_mismatches
    return total

    

# reversal count as one
# this function is for outputs of gies
# A[i,j] = 1, A[j,i] = 0 means edge i->j
# A[i,j] = A[j,i] = 1 means undirected edge i-j
def shd_binary_2(
    A_true: np.ndarray,
    A_hat: np.ndarray
) -> int:
    """
    Structural Hamming Distance (SHD) between two directed graphs.

    If count_reversals_as_one=True:
      - extra edge (FP): +1
      - missing edge (FN): +1
      - reversed edge i->j vs j->i: +1 total (not +2)

    Args:
        A_true: binary ground-truth adjacency (n,n)
        A_hat:  binary estimated adjacency (n,n)
        count_reversals_as_one: whether reversal counts as 1 edit

    Returns:
        nonnegative int
    """
    A_true = np.asarray(A_true).astype(int)
    A_hat = np.asarray(A_hat).astype(int)
    if A_true.shape != A_hat.shape:
        raise ValueError(f"Shape mismatch: {A_true.shape} vs {A_hat.shape}")
    if A_true.ndim != 2 or A_true.shape[0] != A_true.shape[1]:
        raise ValueError("Adjacency matrices must be square (n,n).")

    # For each unordered pair {i,j}, count:
    # 0 if both directions match (both 0/1),
    # 1 if it's a reversal or single-edge mismatch,
    # 2 if they disagree on both directions (very rare if both are DAGs, but handled).
    n = A_true.shape[0]
    total = 0
    for i in range(n):
        for j in range(i + 1, n):
            a_ij, a_ji = A_true[i, j], A_true[j, i]
            h_ij, h_ji = A_hat[i, j], A_hat[j, i]

            # number of directed mismatches for this pair
            edge_mismatches = int(max(h_ij, h_ji) != (a_ij + a_ji)) 
            if edge_mismatches == 0 and (h_ij + h_ji == 1) and ((h_ij==a_ij) or (h_ji==a_ji)):
                total += 1  # reversal
            else:
                total += edge_mismatches
    return total



def tpr_from_weights(W_true: np.ndarray, W_hat: np.ndarray, threshold: float = 1e-8) -> float:
    binarized_true = binarize_adjacency(W_true, threshold=threshold)
    binarized_hat = binarize_adjacency(W_hat, threshold=threshold)
    score = tpr(binarized_true, binarized_hat)
    return score

def shd_from_weights(W_true: np.ndarray, W_hat: np.ndarray, threshold: float = 1e-8, count_reversals_as_one: bool = True) -> int:
    binarized_true = binarize_adjacency(W_true, threshold=threshold)
    binarized_hat = binarize_adjacency(W_hat, threshold=threshold)
    score = shd(binarized_true, binarized_hat, count_reversals_as_one=count_reversals_as_one)
    return score

def f1_score(Lambda_est, Lambda_true):
    est_support = (np.abs(Lambda_est) > 0).astype(int)
    true_support = (np.abs(Lambda_true) > 0).astype(int)
    true_positives = np.sum((est_support == 1) & (true_support == 1))
    false_positives = np.sum((est_support == 1) & (true_support == 0))
    false_negatives = np.sum((est_support == 0) & (true_support == 1))
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    f1_score = 2 * precision * recall / (precision + recall)
    return f1_score

def safe_f1_score(estimate: np.ndarray, truth: np.ndarray) -> float:
    with np.errstate(divide="ignore", invalid="ignore"):
        score = f1_score(estimate, truth)
    return float(score) if np.isfinite(score) else np.nan


def relative_frob_score(estimate: np.ndarray, truth: np.ndarray) -> float:
    denom = float(np.sum(truth**2))
    if denom <= 0:
        return np.nan
    return float(np.sqrt(np.sum((estimate - truth) ** 2) / denom))


def adjacency_to_parent_dict(adjacency: np.ndarray) -> dict[int, list[int]]:
    adjacency = np.asarray(adjacency)
    n_nodes = adjacency.shape[0]
    return {
        child: [parent for parent in range(n_nodes) if adjacency[parent, child] != 0]
        for child in range(n_nodes)
    }


def count_wrong_parents(order, true_dag: dict[int, list[int]]) -> float:
    if order is None:
        return np.nan

    wrong = 0
    seen: set[int] = set()
    for node in np.asarray(order, dtype=int).tolist():
        for parent in true_dag[int(node)]:
            if parent not in seen:
                wrong += 1
        seen.add(int(node))
    return float(wrong)
