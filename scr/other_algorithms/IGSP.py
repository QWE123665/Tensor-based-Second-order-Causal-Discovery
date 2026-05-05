import causaldag as cd 
import numpy as np
from sklearn.linear_model import LinearRegression


def IGSP(X_list, B,alpha_ci = 1e-3, alpha_inv = 1e-3):
    setting_list = []
    for i in range(1,B.shape[1]):
        setting_list.append({"interventions": set(np.where(B[:,i]==0)[0])})
    # -----------------------------
    # CI tester (observational): partial correlation for Gaussian data
    # -----------------------------
    ci_suff = cd.partial_correlation_suffstat(X_list[0])
    ci_tester = cd.MemoizedCI_Tester(cd.partial_correlation_test, ci_suff, alpha=alpha_ci)

    # -----------------------------
    # Invariance tester (obs vs interventions): Gaussian invariance
    # -----------------------------
    inv_suff = cd.gauss_invariance_suffstat(X_list[0], X_list[1:])
    inv_tester = cd.MemoizedInvarianceTester(cd.gauss_invariance_test, inv_suff, alpha=alpha_inv)

    est_dag = cd.igsp(
        setting_list=setting_list,
        nodes=set(range(X_list[0].shape[1])),
        ci_tester=ci_tester,
        invariance_tester=inv_tester,
        depth = 4,
        nruns=10,
        initial_undirected="threshold",
        verbose=False,
    )
    causal_order = est_dag.topological_sort()
    X_obs = X_list[0]
    d = X_obs.shape[1]
    Lambda_= np.zeros((d, d))
    for i in range(d):
        parents = [j for (j, k) in est_dag.arcs if k == i]
        if not parents:
            continue

        reg = LinearRegression(fit_intercept=True)
        reg.fit(X_obs[:, parents], X_obs[:, i])

        for idx, j in enumerate(parents):
            Lambda_[j,i] = reg.coef_[idx]
    return Lambda_, causal_order
