# Tensor-based Second-order Causal Discovery (TSCD)

Causal discovery from interventional data using second-order (covariance) statistics. Given samples from multiple environments — each defined by a known intervention pattern — TSCD recovers a causal order over the variables and an estimate of the linear adjacency matrix.

## Method

TSCD operates on the stack of inverse covariances across environments. For a linear SEM, this stack is a third-order tensor whose structure encodes the causal order. The algorithm peels nodes one at a time:

1. Score each remaining node by the projected norm of its row in the orthonormalized concentration tensor.
2. Among the top candidates, run pairwise correlation tests across intervention contexts (`both intervened`, `i intervened only`, `j intervened only`) to decide adjacency and direction.
3. Schur-complement out the chosen root and repeat.

Two variants are provided:

- **`TSCD`** ([scr/TSCD.py](scr/TSCD.py)) — root selection via stability of projection norm plus pairwise tests.
- **`TSCD_ancestor_search`** ([scr/TSCD_ancestor_search.py](scr/TSCD_ancestor_search.py)) — at each peel, walks upward from top-ranked candidates by repeatedly testing for parents, returning all parent-less nodes in one batch.

After the order is recovered, the adjacency matrix is fit by regressing each node on its predecessors.

Each variant has an **online** counterpart — `TSCD_online` and `TSCD_ancestor_search_online` — that takes per-environment covariance matrices and sample sizes directly instead of raw `X_list`. The online versions are useful when samples are too large to hold in memory or when only summary statistics are available. They recover Lambda via a Cholesky decomposition of the observational covariance in the recovered order.

## Repository layout

```
scr/
  TSCD.py                    # main algorithm (+ TSCD_online: cov_list input)
  TSCD_ancestor_search.py    # ancestor-search variant (+ _online counterpart)
  generate_LSEM.py           # synthetic linear SEM data generation
  myutils.py                 # tensor / regression helpers
  metrics.py                 # SHD, edge-error, etc.
  TSCD_nonlinear.py          # nonlinear SEM extension
  other_algorithms/          # baselines: GES, GIES, PC, IGSP, LiNGAM, NoTears, sort_regress
experiments_script/
  experiment_harness_LSEM.py # sweep over (n_nodes, sample_size, edge_prob, noise) x methods
  harness_helpers.py
  nonlinear_SEM.py           # nonlinear experiment driver
  experiment_analysis_LSEM.ipynb
  scalability.ipynb
```

> Note: the source folder is named `scr/` (not `src/`).

## Installation

Python 3.9+ recommended. Core dependencies:

```bash
pip install numpy scipy scikit-learn pandas
```

For the baselines in `scr/other_algorithms/`, additional packages are required (e.g. `causaldag`, `lingam`, `networkx`, `cdt`). Install as needed by the methods you want to compare against.

## Quick start

```python
import numpy as np
import sys; sys.path.insert(0, "scr")

from generate_LSEM import generate_LSEM_samples_perfect, binary_code_array
from TSCD import TSCD

n_nodes = 10
edge_prob = 0.6
seed = 0
eps_var = np.random.default_rng(seed).uniform(0.1, 1.0, size=n_nodes)

# Intervention design: columns are environments, rows are nodes.
B = binary_code_array(n_nodes, observational=True)
sample_sizes = [500] * B.shape[1]

Lambda_true, X_list, perm, _, _ = generate_LSEM_samples_perfect(
    n_nodes, edge_prob, sample_sizes, B, random_state=seed, eps_var=eps_var,
)

Lambda_est, node_order = TSCD(X_list, B, n_candidates=2)
```

`X_list[i]` is the sample matrix for environment `i`; `B[j, i] == 1` indicates that node `j` is intervened in environment `i`.

If you only have per-environment covariance matrices (e.g. precomputed summaries), use the online variant:

```python
from TSCD import TSCD_online

cov_list = [np.cov(X.T) for X in X_list]
sample_sizes = [X.shape[0] for X in X_list]

Lambda_est, node_order = TSCD_online(cov_list, B, sample_sizes, n_candidates=2)
```

By default the Cholesky decoding uses `cov_list[0]`; pass `cov_obs_idx=i` to use a different environment as the observational reference.

