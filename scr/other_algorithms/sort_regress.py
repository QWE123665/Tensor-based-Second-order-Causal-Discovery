import numpy as np
from utils import _estimate_adjacency_matrix



def _update_cov(cov, root_idx):
    """Update a list of covariance matrices by removing the effect of root_idx."""
    updated_cov_list = []
   
    varx = cov[root_idx, root_idx]
    v = cov[root_idx, :].reshape(-1, 1)
    if varx!=0:
        updated_cov = cov - v @ v.T / varx
    else:
        updated_cov = cov 
    mask = np.delete(np.arange(cov.shape[0]), root_idx)
    updated_cov = updated_cov[np.ix_(mask, mask)]
    return updated_cov,mask



def SortRegress(X):
    """
    SortRegress: estimate adjacency matrix by obtaining causal order from ranking the varianecs
    then do regression.
    """
    p = X.shape[1]
    original_indices = np.arange(p)
    perm = []
    cov = np.cov(X.T)
    while len(perm)<p:
        root_idx = np.argmin(np.diag(cov))
        perm.append(original_indices[root_idx])
        cov, mask = _update_cov(cov, root_idx)
        original_indices = original_indices[mask]

    Lambda_est=_estimate_adjacency_matrix(perm, X)
    return Lambda_est, perm


if __name__ == "__main__":
    print("Testing _update_cov...")
    cov = np.array(
        [
            [1.0, 0.2, 0.0],
            [0.2, 4.0, 0.1],
            [0.0, 0.1, 9.0],
        ]
    )
    updated_cov, mask = _update_cov(cov, 0)
    print(f"Updated covariance shape: {updated_cov.shape}")
    print(f"Mask after removing node 0: {mask}")
    assert updated_cov.shape == (2, 2), f"Expected (2, 2), got {updated_cov.shape}"
    assert np.array_equal(mask, np.array([1, 2])), f"Unexpected mask: {mask}"
    print("_update_cov test passed.")

    print("Testing SortRegress...")
    X = np.array(
        [
            [-1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, -2.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, -3.0],
            [0.0, 0.0, 3.0],
        ]
    )

    captured = {}

    def fake_estimate_adjacency_matrix(perm, data, prior_knowledge=None):
        captured["perm"] = list(perm)
        captured["shape"] = data.shape
        return np.arange(9, dtype=float).reshape(3, 3)

    original = _estimate_adjacency_matrix
    globals()["_estimate_adjacency_matrix"] = fake_estimate_adjacency_matrix
    try:
        Lambda_est, aux = SortRegress(X)
    finally:
        globals()["_estimate_adjacency_matrix"] = original

    print(f"Shape of Lambda_est: {Lambda_est.shape}")
    print(f"Inferred order: {captured['perm']}")
    assert Lambda_est.shape == (3, 3), f"Expected (3, 3), got {Lambda_est.shape}"
    assert captured["perm"] == [0, 1, 2], f"Unexpected order: {captured['perm']}"
    assert captured["shape"] == X.shape, f"Unexpected input shape: {captured['shape']}"
    assert aux is None
    print("SortRegress test passed.")

    print("All tests passed!")
