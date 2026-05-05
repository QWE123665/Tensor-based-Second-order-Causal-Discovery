import numpy as np
from itertools import permutations
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LassoLarsIC
from sklearn.preprocessing import StandardScaler
from scipy.linalg import qr as scipy_qr
from scipy.stats import t, norm
from utils import _estimate_adjacency_matrix,_update_cov_list
try:
    from .utils import *
    from .generate_LSEM import *
except ImportError:
    from utils import *
    from generate_LSEM import *



# ---------- Test 1: null hypothesis correlation =0 ----------
def test_nonzero_correlation(n, r):
    # t statistic
    t_stat = r * np.sqrt((n - 2) / (1 - r**2))

    # two-sided p-value
    p_value = 2 * (1 - t.cdf(abs(t_stat), df=n-2))

    return t_stat, p_value




# ---------- Test 2: null hypothesis |correlation|>epsilon ----------
def test_small_correlation(n, r, epsilon=0.1, alpha=0.1):
    # Fisher transform
    z_r = 0.5 * np.log((1 + r) / (1 - r))
    z_eps = 0.5 * np.log((1 + epsilon) / (1 - epsilon))
    SE = 1 / np.sqrt(n - 3)

    # two one-sided tests
    Z1 = (z_r - (-z_eps)) / SE   # test rho > -epsilon
    Z2 = (z_r - z_eps) / SE      # test rho < epsilon
    z_crit = norm.ppf(1 - alpha)
    reject_lower = Z1 > z_crit
    reject_upper = Z2 < -z_crit
    equivalence = reject_lower and reject_upper
    return {
        "Z1": Z1,
        "Z2": Z2,
        "equivalent": equivalence
    }


# ---------- Combined decision ----------

def analyze_correlation(n, r, epsilon=0.1, alpha=0.1):
    t_stat, p_val = test_nonzero_correlation(n, r)
    eq_test = test_small_correlation(n, r, epsilon, alpha)

    if p_val < alpha:
        decision = "NONZERO correlation"
        decision_num = 1
    elif eq_test["equivalent"]:
        decision = "EFFECTIVELY ZERO correlation"
        decision_num = 0
    else:
        decision = "INCONCLUSIVE"
        decision_num = -1

    return {
        "r": r,
        "n": n,
        "t_stat": t_stat,
        "p_value": p_val,
        "equivalence": eq_test["equivalent"],
        "decision": decision,
        "decision_num": decision_num
    }







def analyze_pair(i, j, cov_list, B, sample_sizes, verbose=False, alpha=0.05, epsilon=0.1):
    # corr_list list of correlations between two nodes size k
    # interventions size 2 by k 

    k = B.shape[1]
    interventions = B[[i,j],:]
    corr_list = [cov_list[l][i,j]/np.sqrt(cov_list[l][i,i]*cov_list[l][j,j]) for l in range(k)]
    both_exist = [l for l in range(k) if (interventions[0,l] & interventions[1,l])]
    intervened_not = [l  for l in range(k) if (~interventions[0,l] & interventions[1,l]) ]
    not_intervened = [l for l in range(k) if (interventions[0,l] & ~interventions[1,l]) ]

    # Aggregate evidence across all context groups so later checks do not overwrite earlier evidence.
    adjacency_evidence_true = 0
    adjacency_evidence_false = 0
    direction_score = {i: 0, j: 0}

    both_exist_decisions = [analyze_correlation(sample_sizes[l], corr_list[l], epsilon=epsilon, alpha=alpha)['decision_num'] for l in both_exist]

    if 1 in both_exist_decisions and 0 in both_exist_decisions and both_exist_decisions[0]==1:
        adjacency_evidence_false += 1

    intervened_not_decisions = [analyze_correlation(sample_sizes[l], corr_list[l], epsilon=epsilon, alpha=alpha)['decision_num'] for l in intervened_not]
    if 1 in intervened_not_decisions and 0 not in intervened_not_decisions and both_exist_decisions[0]==1:
        adjacency_evidence_true += 1
        direction_score[i] += 1
        direction_score[j] -= 1
    elif 1 in intervened_not_decisions and 0 in intervened_not_decisions and both_exist_decisions[0]==1:
        adjacency_evidence_false += 1
        direction_score[i] += 1
        direction_score[j] -= 1
    elif 1 not in intervened_not_decisions and 0 in intervened_not_decisions and both_exist_decisions[0]==1:
        direction_score[i] -= 1
    
    not_intervened_decisions = [analyze_correlation(sample_sizes[l], corr_list[l], epsilon=epsilon, alpha=alpha)['decision_num'] for l in not_intervened]
    if 1 in not_intervened_decisions and 0 not in not_intervened_decisions and both_exist_decisions[0]==1:
        adjacency_evidence_true += 1
        direction_score[j] += 1
        direction_score[i] -= 1
    elif 1 in not_intervened_decisions and 0 in not_intervened_decisions and both_exist_decisions[0]==1:
        adjacency_evidence_false += 1
        direction_score[j] += 1
        direction_score[i] -= 1
    elif 1 not in not_intervened_decisions and 0 in not_intervened_decisions and both_exist_decisions[0]==1:
        direction_score[j] -= 1

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
    



def _select_root_by_stability_proj_norm_simplify(
    cov_list,
    T_U,
    B,
    original_indices,
    sample_sizes,
    n_candidates=3,
    verbose=False,
    alpha=0.05,
    epsilon=0.1,
):
    p = cov_list[0].shape[0]
    n_candidates = max(1, min(int(n_candidates), p))
    possible_idxs = np.arange(p)

    proj_norms = np.array([
        np.linalg.norm(
            np.einsum(
                "ijl,jl->i",
                T_U,
                np.eye(T_U.shape[0])[[i], :][0:1].T * B[possible_idxs[i], :],
            )
        ) / np.linalg.norm(B[possible_idxs[i], :])
        for i in range(T_U.shape[0])
    ])

    if p <= n_candidates:
        close_node_ids = possible_idxs
    else:
        top_idx = np.argpartition(proj_norms, -n_candidates)[-n_candidates:]
        order = np.argsort(proj_norms[top_idx])[::-1]
        close_node_ids = top_idx[order]

    if verbose:
        print(f"close_nodes & proj_norm {[(original_indices[i],proj_norms[i]) for i in close_node_ids]}")
        print(proj_norms)

    pair_scores = {idx: 0 for idx in close_node_ids}
    for a_pos in range(len(close_node_ids)):
        i = close_node_ids[a_pos]
        for b_pos in range(a_pos + 1, len(close_node_ids)):
            j = close_node_ids[b_pos]
            _, earlier, latter = analyze_pair(
                i,
                j,
                cov_list,
                B,
                sample_sizes,
                verbose=verbose,
                alpha=alpha,
                epsilon=epsilon,
            )
            if earlier is not None:
                pair_scores[earlier] += 1
            if latter is not None:
                pair_scores[latter] -= 1
            if verbose:
                print(f"compare {i} & {j} {pair_scores}")

    close_nodes = [idx for idx, score in pair_scores.items() if score == len(close_node_ids)-1]
    if len(close_nodes)==0:
        close_nodes = [idx for idx, score in pair_scores.items() if score >=0]
    if len(close_nodes) == 0:
        best_idx = int(np.argmax(proj_norms))
    elif len(close_nodes) == 1:
        best_idx = int(close_nodes[0])
    else:
        best_idx = int(max(close_nodes, key=lambda idx: proj_norms[idx]))

    return best_idx


def causal_order_from_proj_norms(
    B,
    covlist,
    T,
    sample_sizes,
    n_candidates=3,
    verbose=False,
    alpha=0.05,
    epsilon=0.1,
):
    n, _ = B.shape
    perm = []
    original_indices = np.arange(n)

    while len(perm) < n:
        T_U = orthonormal_from_householder_qr(T)
        root_idx = _select_root_by_stability_proj_norm_simplify(
            covlist,
            T_U,
            B[original_indices, :],
            original_indices,
            sample_sizes,
            n_candidates=n_candidates,
            verbose=verbose,
            alpha=alpha,
            epsilon=epsilon,
        )

        if verbose:
            print(f"=====================found node {original_indices[root_idx]}=========================")
            print(f"======================progress: {round(len(perm)/n*100)}% ===============================")

        covlist, mask = _update_cov_list(covlist, root_idx)
        perm.append(original_indices[root_idx])
        T = T[np.ix_(mask, mask)]

        original_indices = original_indices[mask]
    return perm



def TSCD(X_list, B, 
        n_candidates = 3, pthreshold = 0.01, 
        verbose = False, only_obs = False,
        alpha = 0.05, epsilon = 0.1):
    n,k = B.shape

    covlist = [np.cov(X.T) for X in X_list]
    sample_sizes = [X.shape[0] for X in X_list]
    M_list = concentration_list(X_list,B)
    T = np.stack(M_list,axis = -1)
    

    node_permutation = causal_order_from_proj_norms(B, covlist, T, sample_sizes, n_candidates = n_candidates, verbose = verbose, alpha = alpha, epsilon = epsilon)
    if only_obs:
        Lambda_est = _estimate_adjacency_matrix(node_permutation, X_list[0])
    else:
        Lambda_est = improve_Lambda_causal_order_regression_all_pure(X_list, B, node_permutation)
    
    return Lambda_est, node_permutation





if __name__ == "__main__":
    import numpy as np
    try:
        from .generate_LSEM import generate_LSEM_samples_perfect, generate_LSEM_samples_soft, binary_code_array
    except ImportError:
        from generate_LSEM import generate_LSEM_samples_perfect, generate_LSEM_samples_soft, binary_code_array

    def count_wrong_parents(order, true_Lambda):
        """
        Count how many true parents appear after their child in the recovered order.
        """
        wrong = 0
        seen = set()
        for node in order:
            for parent in np.where(true_Lambda[node,:]!=0)[0]:
                if parent not in seen:
                    print(f'node {node} is too early')
                    wrong += 1
            seen.add(node)  # fix: add node (not parent)
        return wrong
    
    # # Test our_method_simple
    print("Testing our_method_simple_concentration...")
    n_nodes = 10
    edge_prob = 0.6
    seed = np.random.randint(0,1000)
    eps_var = np.random.uniform(0.1,1,size = n_nodes)
 
    B = binary_code_array(n_nodes, observational=True)
    sample_sizes = [500] * B.shape[1]
    Lambda_true, X_list, perm,_,_ = generate_LSEM_samples_perfect(n_nodes, edge_prob, sample_sizes, B, random_state=seed,eps_var = eps_var)
    Lambda_est, node_permutation = TSCD(X_list, B, n_candidates = 2)
    print(count_wrong_parents(node_permutation,Lambda_true), relFrob_error(Lambda_est,Lambda_true))
