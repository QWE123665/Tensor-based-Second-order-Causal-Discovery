import numpy as np
from itertools import permutations
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LassoLarsIC
from sklearn.preprocessing import StandardScaler
from scipy.linalg import qr as scipy_qr
from scipy.stats import t, norm
from myutils import _update_cov_list, lambda_from_causal_order_regression
from TSCD import analyze_correlation
try:
    from .myutils import *
    from .generate_LSEM import *
except ImportError:
    from myutils import *
    from generate_LSEM import *






def analyze_parent(i, j, cov_list, B, sample_sizes, verbose=False, alpha=0.05, epsilon=0.1):
    # corr_list list of correlations between two nodes size k
    # interventions size 2 by k 

    k = B.shape[1]
    interventions = B[[i,j],:]
    corr_list = [cov_list[l][i,j]/np.sqrt(cov_list[l][i,i]*cov_list[l][j,j]) if (cov_list[l][i,i] != 0 and cov_list[l][j,j] != 0) else 0 for l in range(k)]
    both_exist = [l for l in range(k) if (interventions[0,l] & interventions[1,l])]
    intervened_not = [l  for l in range(k) if (~interventions[0,l] & interventions[1,l]) ]
    not_intervened = [l for l in range(k) if (interventions[0,l] & ~interventions[1,l]) ]

    # Aggregate evidence across all context groups so later checks do not overwrite earlier evidence.
    adjacency_evidence_true = 0
    adjacency_evidence_false = 0
    direction_score = {i: 0, j: 0}

    
    both_exist_decisions = [analyze_correlation(sample_sizes[l], corr_list[l], epsilon=epsilon, alpha=alpha)['decision_num'] for l in both_exist]

    # if 1 in both_exist_decisions and 0 in both_exist_decisions and both_exist_decisions[0]==1:
    #     adjacency_evidence_false += 1

    intervened_not_decisions = [analyze_correlation(sample_sizes[l], corr_list[l], epsilon=epsilon, alpha=alpha)['decision_num'] for l in intervened_not]
    if 1 in intervened_not_decisions and 0 not in intervened_not_decisions and both_exist_decisions[0]==1:
        adjacency_evidence_true += 1
        direction_score[i] += 1
        direction_score[j] -= 1
    elif 1 in intervened_not_decisions and 0 in intervened_not_decisions and both_exist_decisions[0]==1:
        adjacency_evidence_false += 1
        direction_score[i] += 1
        direction_score[j] -= 1
    # elif 1 not in intervened_not_decisions and 0 in intervened_not_decisions and both_exist_decisions[0]==1:
    #     direction_score[i] -= 1
    
    not_intervened_decisions = [analyze_correlation(sample_sizes[l], corr_list[l], epsilon=epsilon, alpha=alpha)['decision_num'] for l in not_intervened]
    if 1 in not_intervened_decisions and 0 not in not_intervened_decisions and both_exist_decisions[0]==1:
        adjacency_evidence_true += 1
        direction_score[j] += 1
        direction_score[i] -= 1
    elif 1 in not_intervened_decisions and 0 in not_intervened_decisions and both_exist_decisions[0]==1:
        adjacency_evidence_false += 1
        direction_score[j] += 1
        direction_score[i] -= 1
    # elif 1 not in not_intervened_decisions and 0 in not_intervened_decisions and both_exist_decisions[0]==1:
    #     direction_score[j] -= 1

    # Final adjacency decision from aggregated evidence.
    if adjacency_evidence_true > adjacency_evidence_false:
        adjacency_flag = True
    elif adjacency_evidence_false > 0:
        adjacency_flag = False
    else:
        adjacency_flag = None

    # Final order decision from aggregated directional score.
    if direction_score[i] > direction_score[j]:
        earlier_idx, latter_idx = i, j
    elif direction_score[j] > direction_score[i]:
        earlier_idx, latter_idx = j, i
    else:
        earlier_idx, latter_idx = None, None

    if verbose:
        print(f'pair ({i}, {j})')
        print(f'  both_exist idx={both_exist} decisions={both_exist_decisions} corr={[corr_list[l] for l in both_exist]}')
        print(f'  i_intervened_j_not idx={intervened_not} decisions={intervened_not_decisions} corr={[corr_list[l] for l in intervened_not]}')
        print(f'  i_not_j_intervened idx={not_intervened} decisions={not_intervened_decisions} corr={[corr_list[l] for l in not_intervened]}')
        print(f'  evidence: adj_true={adjacency_evidence_true}, adj_false={adjacency_evidence_false}, direction_score={direction_score}')
        print(f'  result: adjacency_flag={adjacency_flag}, earlier_idx={earlier_idx}, latter_idx={latter_idx}')

    return adjacency_flag, earlier_idx, latter_idx


def find_ancestors(
    node,
    cov_list,
    B,
    sample_sizes,
    candidates=None,
    verbose=False,
    alpha=0.05,
    epsilon=0.1,
    strict=True,
):
    """Identify direct parents of ``node`` via pairwise tests.

    Calls ``analyze_pair(node, j)`` for each ``j`` in ``candidates`` and keeps
    those ordered earlier than ``node`` and flagged as adjacent.

    Parameters
    ----------
    node : int
        Index of the target child.
    cov_list, B, sample_sizes : same as ``analyze_pair``.
    candidates : iterable of int, optional
        Nodes to test. Defaults to all nodes except ``node``.
    strict : bool
        If True, require ``adjacency_flag is True``. If False, also accept
        ``adjacency_flag is None`` (inconclusive adjacency but j is earlier).

    Returns
    -------
    parents : list of int
        Indices that test as direct parents of ``node``.
    """
    p = cov_list[0].shape[0]
    if candidates is None:
        candidates = [j for j in range(p) if j != node]

    parents = []
    for j in candidates:
        if j == node:
            continue
        adjacency_flag, earlier, latter = analyze_parent(
            node, j, cov_list, B, sample_sizes,
            verbose=False, alpha=alpha, epsilon=epsilon,
        )
        is_earlier = (earlier == j and latter == node)
        if strict:
            is_adjacent = (adjacency_flag is True)
        else:
            is_adjacent = (adjacency_flag is not False)
        if is_earlier:
            parents.append(j)
        if verbose:
            print(f"  candidate {j}: adj={adjacency_flag} earlier={earlier} "
                  f"latter={latter} -> parent={j in parents}")

    return parents


def select_root_via_ancestor_search(
    candidates,
    cov_list,
    B,
    sample_sizes,
    search_pool=None,
    verbose=False,
    alpha=0.05,
    epsilon=0.1,
    strict=True,
    node_labels=None,
):
    """Collect all parent-less nodes by walking upward through ``find_direct_parents``.

    Processes ``candidates`` in order. For each one, runs ``find_direct_parents``;
    parent-less nodes are accumulated as roots. When a node has parents, those
    parents (restricted to ``search_pool`` if given) are appended to the queue
    so they get tested too. The walk does not early-exit — it continues until
    every reachable node has been examined.

    Parameters
    ----------
    candidates : iterable of int
        Initial candidate indices.
    search_pool : iterable of int, optional
        Restrict ``find_direct_parents`` to test only these indices as
        potential parents. Defaults to all nodes.

    Returns
    -------
    roots : list of int
        Every visited node that tested as having no direct parents (in
        visit order). Empty if none were found.
    visited : list of int
        All nodes examined during the search (in visit order).
    """
    queue = list(candidates)
    visited = []
    seen = set()
    pool = None if search_pool is None else list(search_pool)
    roots = []

    while queue:
        node = queue.pop(0)
        if node in seen:
            continue
        seen.add(node)
        visited.append(node)

        parents = find_ancestors(
            node, cov_list, B, sample_sizes,
            candidates=pool,
            verbose=False, alpha=alpha, epsilon=epsilon, strict=strict,
        )

        if verbose:
            if node_labels is not None:
                print(f"  visit node {node_labels[node]}: "
                      f"parents={[node_labels[p] for p in parents]}")
            else:
                print(f"  visit node {node}: parents={parents}")

        if len(parents) == 0:
            roots.append(node)
        else:
            for p in parents:
                if p not in seen and p not in queue:
                    queue.append(p)

    if verbose and len(roots) == 0:
        print("  no parent-less node found")

    return roots, visited





def causal_order_from_parent_search(
    B,
    covlist,
    T,
    sample_sizes,
    n_candidates=3,
    pool_multiplier=10,
    verbose=False,
    alpha=0.05,
    epsilon=0.1,
    strict=True,
):
    """Recover a causal order by parent-search at each peeling step.

    At every iteration: rank remaining nodes by ``proj_norm``; take the top
    ``n_candidates`` as the search seeds and the top
    ``pool_multiplier * n_candidates`` as the pool of admissible parents;
    call ``select_root_via_parent_search``, which returns *all* parent-less
    nodes it finds. Schur-complement out every such root in one pass before
    recomputing proj_norms.
    """
    n, _ = B.shape
    perm = []
    original_indices = np.arange(n)

    while len(perm) < n:
        p = covlist[0].shape[0]
        T_U = orthonormal_from_householder_qr(T)

        if verbose:
            print('T_u computed')
        B_local = B[original_indices, :]

        proj_norms = np.array([
            np.linalg.norm(
                np.einsum(
                    "ijl,jl->i",
                    T_U,
                    np.eye(T_U.shape[0])[[i], :][0:1].T * B_local[i, :],
                )
            ) / np.linalg.norm(B_local[i, :])
            for i in range(p)
        ])

        # if proj_norms = 0 it means the node's variance is 0 so also treated as stable 
        proj_norms = np.where(proj_norms == 0, 1, proj_norms)

        n_seed = max(1, min(int(n_candidates), p))
        n_pool = max(n_seed, min(int(pool_multiplier) * int(n_candidates), p))
        order_desc = np.argsort(proj_norms)[::-1]
        seeds = order_desc[:n_seed].tolist()
        pool = order_desc[:n_pool].tolist()

        if verbose:
            print(f"proj_norms (top {n_pool}): "
                  f"{[(int(original_indices[i]), float(proj_norms[i])) for i in pool]}")

        roots, visited = select_root_via_ancestor_search(
            seeds,
            covlist,
            B_local,
            sample_sizes,
            search_pool=pool,
            verbose=verbose,
            alpha=alpha,
            epsilon=epsilon,
            strict=strict,
            node_labels=original_indices,
        )

        if len(roots) == 0:
            # noisy cycle — fall back to highest proj_norm node
            roots = [int(order_desc[0])]
            if verbose:
                print(f"  fallback: peeling top proj_norm node {original_indices[roots[0]]}")

        # Order roots within this batch by row-sum of B (largest first):
        # nodes not intervened in more contexts go earlier in perm.
        roots_sorted = sorted(set(roots), key=lambda r: -B_local[r, :].sum())
        roots_global = [int(original_indices[r]) for r in roots_sorted]

        if verbose:
            print(f"=====================found roots {roots_global} "
                  f"(visited={[int(original_indices[v]) for v in visited]})=====================")
            print(f"======================progress: {round(len(perm)/n*100)}% ===============================")

        # Peel one at a time — local indices shift after each Schur update.
        for g in roots_global:
            local = int(np.where(original_indices == g)[0][0])
            covlist, mask = _update_cov_list(covlist, local)
            perm.append(g)
            T = T[np.ix_(mask, mask)]
            original_indices = original_indices[mask]

    return perm


def TSCD_ancestor_search(
    X_list,
    B,
    n_candidates=3,
    pool_multiplier=10,
    threshold=0.1,
    ridge=1e-10,
    verbose=False,
    alpha=0.05,
    epsilon=0.1,
    strict=True,
):
    """End-to-end LSEM recovery via parent-search ordering plus regression fit.

    Wraps ``causal_order_from_parent_search`` (to recover the causal order)
    and ``lambda_from_causal_order_regression`` (to fit ``Lambda`` from the
    observational covariance, no Cholesky).

    Returns
    -------
    Lambda_est : (n, n) ndarray
        Estimated adjacency matrix.
    node_permutation : list of int
        Recovered causal order (root first).
    """
    cov_list = [np.cov(X.T) for X in X_list]
    sample_sizes = [X.shape[0] for X in X_list]
    M_list = concentration_list(X_list, B)
    T = np.stack(M_list, axis=-1)

    node_permutation = causal_order_from_parent_search(
        B,
        cov_list,
        T,
        sample_sizes,
        n_candidates=n_candidates,
        pool_multiplier=pool_multiplier,
        verbose=verbose,
        alpha=alpha,
        epsilon=epsilon,
        strict=strict,
    )
    Lambda_est = lambda_from_causal_order_regression(
        cov_list[0], node_permutation, threshold=threshold, ridge=ridge,
    )
    return Lambda_est, node_permutation




