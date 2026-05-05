import sys
import time
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent / "scr"))

from nonlinear import (
    SEED,
    SEM_perfect_intervention,
    generate_DAG,
    generate_binary_adjacency_matrix,
    learn_graph_parallel,
)
from generate_LSEM import binary_code_single_intervene

start = time.perf_counter()
n_nodes = 10
seed = SEED
edge_prob = 0.8
sample_size_per_env = 2000

DAG = generate_DAG(num_nodes=n_nodes, edge_probability=edge_prob, random_state=seed)
intervention_matrix = binary_code_single_intervene(n_nodes)
sample_sizes = [sample_size_per_env] * intervention_matrix.shape[1]


samples, f_lists, f_details = SEM_perfect_intervention(
    DAG, intervention_matrix, sample_sizes, random_state=seed
)




def count_wrong_parents(order, true_DAG):
    """
    Count how many true parents appear after their child in the recovered order.
    """
    wrong = 0
    seen = set()
    for node in order:
        for parent in true_DAG[node]:
            if parent not in seen:
                wrong += 1
        seen.add(node)  # fix: add node (not parent)
    return wrong
nodes_sorted = sorted(samples[0].keys())
var_per_node = np.array([np.var(samples[0][k]) for k in nodes_sorted])
order_by_var = [nodes_sorted[i] for i in np.argsort(var_per_node)]
print('sortvar', order_by_var,'count_error',count_wrong_parents(order_by_var,DAG))
perm, graph_hat, _ = learn_graph_parallel(samples, intervention_matrix, verbose=False, max_workers = 5)
graph_true = generate_binary_adjacency_matrix(DAG)

shd = int(np.abs(graph_true - graph_hat).sum())  # edge mismatches (directed)
n_true_edges = int(np.abs(graph_true).sum())
edge_error_rate = shd / n_true_edges if n_true_edges > 0 else 0.0

print("Recovered graph:\n", graph_hat.astype(int))
print("True graph:\n", DAG)
print(f"Learned causal order: {perm}")
print(f"SHD-like edge mismatch count: {shd}")
print(f"Edge error rate (mismatches / #true_edges): {edge_error_rate:.4f}")
print(f"Order violations (#parents appearing after child): {count_wrong_parents(perm, DAG)}")
elapsed = time.perf_counter()-start
print(f"Runtime {elapsed}")