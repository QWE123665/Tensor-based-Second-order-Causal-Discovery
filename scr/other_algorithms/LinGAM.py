from lingam import DirectLiNGAM, ICALiNGAM

import numpy as np


def _zscore(X): # standardization
    X = np.asarray(X, float)
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, ddof=1, keepdims=True)
    sd[sd == 0] = 1.0
    return (X - mu) / sd


def fit_lingam_from_lsem(X,random_state=None, standardize=False, method="ica", **estimator_kwargs):
    """
    fit a LiNGAM estimator ('direct' or 'ica'), return (Lambda_hat, causal_order).
    
    """

    # Seed handling
    if isinstance(random_state, np.random.Generator):
        seed = int(random_state.integers(2**31 - 1))
    elif isinstance(random_state, (int, np.integer)):
        seed = int(random_state)
    else:
        seed = None

    # Pick estimator
    if method.lower() in {"direct", "directlingam"}:
        Model = DirectLiNGAM
    elif method.lower() in {"ica", "icalingam"}:
        Model = ICALiNGAM
    else:
        raise ValueError(f"Unknown method: {method!r}. Use 'direct' or 'ica'.")

    model = Model(random_state=seed, **estimator_kwargs)

    if standardize:
        X = _zscore(X)

    model.fit(X)

    Lamb_hat = np.asarray(model.adjacency_matrix_, dtype=float)  # j -> i

    causal_order = getattr(model, "causal_order_", None)

    return Lamb_hat, causal_order


def lingam_ica(X, random_state=None, standardize=False, **estimator_kwargs):
    return fit_lingam_from_lsem(X, random_state=random_state, standardize=standardize, method="ica", **estimator_kwargs)

def lingam_direct(X, random_state=None, standardize=False, **estimator_kwargs):
    return fit_lingam_from_lsem(X, random_state=random_state, standardize=standardize, method="direct", **estimator_kwargs)


def lingam_from_interventions(X_list, B, random_state=None, lingam_method="ica"):
    """
    Estimate overall graph from data across contexts. Uses LINGAM to estimate Lambda_hat for each context,
    then averages rows across corresponding unintervened contexts

    Can choose lingam_method as "ica" or "direct"

    Returns: final_Lambda_hat, None 
    (to make return value format consistent with (Lambda_hat, causal_order) )
    """

    if lingam_method == "ica":
        method = lingam_ica
    elif lingam_method == "direct":
        method = lingam_direct
    else:
        raise ValueError("lingam_method must be 'ica' or 'direct'")

    Lamb_hat_list = []
    for X in X_list:
        Lamb_hat, causal_order = method(X, random_state=random_state)
        Lamb_hat_list.append(Lamb_hat)

    # For each node, average its row over contexts where it is not intervened.
    N = B.shape[0]
    final_Lambda_hat = np.zeros((N, N))
    for i in range(N):
        valid_rows = [Lamb_hat_list[j][i, :] for j in range(len(X_list)) if B[i, j] == 1]
        if len(valid_rows) > 0:
            final_Lambda_hat[i, :] = np.mean(valid_rows, axis=0)

    return final_Lambda_hat, None

def lingam_from_interventions_ica(X_list, B, random_state=None):
    return lingam_from_interventions(X_list, B, random_state, "ica")

def lingam_from_interventions_direct(X_list, B, random_state=None):
    return lingam_from_interventions(X_list, B, random_state, "direct")