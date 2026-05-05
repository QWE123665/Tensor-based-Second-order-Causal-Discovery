from generate_LSEM  import *
import numpy as np
from collections import deque
import math
import torch.nn as nn
import torch
import torch.optim as optim
from sklearn.model_selection import train_test_split
import os
import random as rn
from concurrent.futures import ThreadPoolExecutor, as_completed
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
np.random.seed(SEED)
rn.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False





def gaussian_noise_generator_intervention(intervention_matrix, sizes,var_low=0.3, var_high=0.7, random_state=None):
    """
    Returns noise_generator(size, node=None) that outputs Gaussian noise of shape `size`.
    Each node gets a fixed variance sampled once from Uniform[var_low, var_high].
    If node is None, a fresh variance is sampled each call.
    """
    if var_low <= 0 or var_high <= 0 or var_low > var_high:
        raise ValueError("Require 0 < var_low <= var_high.")

    seed_rng = np.random.default_rng(random_state)
    
    p,k = intervention_matrix.shape
    noise = {}
    for node in range(p):
        var = float(seed_rng.uniform(var_low, var_high))

        noise[node] = []
        for i in range(k):
            size = sizes[i]
            if intervention_matrix[node,i]==1:
                # print(f'node {node}, context {i} is not intervened, using variance {var:.4f}')
                noise[node].append(seed_rng.normal(loc=0.0, scale=np.sqrt(var), size=size))
            else:
                var_alter = float(seed_rng.uniform(var_low, var_high))
                # print(f'node {node}, context {i} is intervened, using variance {var_alter:.4f}')
                noise[node].append(seed_rng.normal(loc=0.0, scale=np.sqrt(var_alter), size=size))

    return noise


def random_nonlinear_function(k, random_state=None):
    """
    Generate a random nonlinear function f: R^k -> R.

    Returns
    -------
    f : callable
        Function that maps x (..., k) -> y (...,)
    params : dict
        Randomly sampled parameters used to define f.
    """
    rng = np.random.default_rng(random_state)

    # Random feature count and parameters
    m = rng.integers(3, 8)  # number of hidden nonlinear units
    W = rng.normal(0, 1, size=(m, k))*np.sqrt(6/(m+k))
    b = rng.normal(0, 1, size=m)*np.sqrt(6/m)
    a = rng.normal(0, 1, size=m)/np.sqrt(6/m)

    # Randomly choose nonlinearities per unit
    acts = rng.choice(["tanh", "relu","square"], size=m, replace=True)

    def f(x):
        x = np.asarray(x, dtype=float)
        if x.shape[-1] != k:
            raise ValueError(f"Expected last dimension {k}, got {x.shape[-1]}")


        z = np.maximum(np.minimum(x @ W.T + b,10),-10) # (..., m)
        h = np.empty_like(z)

        for j, act in enumerate(acts):
            if act == "tanh":
                h[..., j] = np.tanh(z[..., j])
            elif act == "relu":
                h[..., j] = np.max(z[..., j],0)
            elif act == "square":
                h[...,j] = z[..., j]**2

        y = h @ a
        return y

    params = {"W": W, "b": b, "a": a, "acts": acts}
    return f, params


def generate_DAG(num_nodes, edge_probability, random_state=None, permute_nodes=True):
    """
    Generate a random DAG with edge j -> i (in base order), then optionally
    permute node labels. Returns dict {node: [parents]}.
    """
    rng = np.random.default_rng(random_state)

    # Base DAG on order 0..num_nodes-1
    base_DAG = {i: [] for i in range(num_nodes)}
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if rng.uniform() < edge_probability:
                base_DAG[j].append(i)

    if not permute_nodes:
        return base_DAG

    # Permute node labels
    perm = rng.permutation(num_nodes)           # old -> new label mapping
    DAG = {int(perm[i]): [] for i in range(num_nodes)}

    for child_old, parents_old in base_DAG.items():
        child_new = int(perm[child_old])
        DAG[child_new] = [int(perm[p]) for p in parents_old]

    return DAG


def generate_binary_adjacency_matrix(DAG):
    """
    DAG: dict {node: [parent1, parent2, ...]}
    Returns A where A[i, j] = 1 iff j is a parent of i.
    """
    nodes = sorted(DAG.keys())
    n = len(nodes)
    node_to_idx = {node: idx for idx, node in enumerate(nodes)}

    A = np.zeros((n, n), dtype=int)
    for child, parents in DAG.items():
        i = node_to_idx[child]
        for p in parents:
            j = node_to_idx[p]
            A[i, j] = 1
    return A


def topological_sort_dag(DAG):
    """
    DAG format: {node: [parent1, parent2, ...]}
    Returns a topological order of nodes.
    """
    children = {u: [] for u in DAG}
    indegree = {u: 0 for u in DAG}

    for u, parents in DAG.items():
        indegree[u] = len(parents)
        for p in parents:
            if p not in DAG:
                raise ValueError(f"Parent {p!r} of node {u!r} is not in DAG keys.")
            children[p].append(u)

    q = deque([u for u in DAG if indegree[u] == 0])
    order = []

    while q:
        u = q.popleft()
        order.append(u)
        for v in children[u]:
            indegree[v] -= 1
            if indegree[v] == 0:
                q.append(v)

    if len(order) != len(DAG):
        raise ValueError("DAG contains a cycle.")

    return order



def SEM_perfect_intervention(DAG, intervention_matrix, sample_sizes, noise_generator=gaussian_noise_generator_intervention,
            f_generator=random_nonlinear_function, random_state=None, DAG_sorted=True):
    """
    Generate nonlinear SEM samples under multiple perfect-intervention contexts.

    This function simulates data context-by-context from a parent-list DAG. Structural
    functions are shared across contexts, while noise is generated per node/context,
    with replaced noise for intervened nodes.

    Graph convention
    ----------------
    DAG is a dict of the form `{node: [parent1, parent2, ...]}` and must be acyclic.

    Intervention convention
    -----------------------
    `intervention_matrix` has shape `(num_nodes, num_contexts)`.
    Entry `intervention_matrix[node, context]` means:
    - `1`: node is *not* intervened in this context.
    - `0`: node is intervened in this context.
    
    The first context is expected to be observational (typically an all-ones column).

    Parent-masking rule used in simulation
    --------------------------------------
    When generating node `v` in context `c`, each parent `p` contributes:
    - `sample[p]` if `intervention_matrix[p, c] == 1`
    - `0` vector otherwise
    This removes the effect of intervened parent nodes from downstream mechanisms.

    Parameters
    ----------
    DAG : dict
        Parent-list DAG: `{node: [parent1, parent2, ...]}`.
    intervention_matrix : np.ndarray, shape (num_nodes, num_contexts)
        Binary intervention indicators for each node/context.
    sample_sizes : int or list[int]
        Number of samples per context. If int, the same size is used for all contexts.
    noise_generator : callable
        Noise factory, same interface as in `SEM()`. Called once, then used to draw
        per-node noise vectors.
    f_generator : callable
        Structural-function factory, same interface as in `SEM()`. Called once per node.
        Generated mechanisms are reused across contexts.
    random_state : int | np.random.Generator | None
        Seed or RNG for reproducible simulation.
    DAG_sorted : bool
        If True, assumes DAG keys are already in topological order.
        If False, computes topological order with `topological_sort_dag`.

    Returns
    -------
    samples : list[dict]
        One dict per context; each dict maps `node -> np.ndarray` of simulated values.
    f_list : dict
        Node-wise structural mechanisms used in all contexts: `{node: callable}`.
    f_details : dict
        Node-wise mechanism metadata returned by `f_generator`.
    """

    num_nodes = len(DAG)
    num_contexts = intervention_matrix.shape[1]

    # Basic shape validation for intervention matrix.
    if intervention_matrix.shape != (num_nodes, num_contexts):
        raise ValueError(f"intervention_matrix should have shape ({num_nodes}, {num_contexts}), got {intervention_matrix.shape})")

    # Normalize sample_sizes to a per-context list.
    if isinstance(sample_sizes, int):
        sample_sizes = [sample_sizes] * num_contexts
    elif len(sample_sizes) != num_contexts:
        raise ValueError(f"sample_sizes should have length {num_contexts}, got {len(sample_sizes)}")

    # Determine generation order: either user-provided DAG order or computed topo order.
    if DAG_sorted:
        order = list(DAG.keys())
    else:
        order = topological_sort_dag(DAG)
    
    
    samples = []
    f_list = {}
    f_details = {}

    # Draw base noise in one concatenated vector per node, then split by context.
    # For intervened contexts, redraw node noise separately.
    
    
    noise = noise_generator(intervention_matrix, sample_sizes, var_low=0.3, var_high=0.7, random_state=random_state)
    # print(f'noise[0] len{len(noise[0])}')
    f_list = {}
    f_details = {}
    rng = np.random.default_rng(random_state)
    for node in order:
        # One structural function per node, reused across contexts.
        node_seed = int(rng.integers(0, 2**32 - 1))
        f,details = f_generator(len(DAG[node]), random_state=node_seed)
        f_list[node] = f
        f_details[node] = details
        # for i,data in enumerate(noise[node]):
        #     if intervention_matrix[node,i]!=1:
        #         print(f'nonroot node {node}, std {np.std(data)}')
    

    
    for i in range(num_contexts):
        sample = {}
        for node in order:
            parents = DAG[node]

            # Build node inputs from parent samples, masking intervened parents to zero.
            if parents:
                parent_values = np.column_stack([
                    sample[p] if intervention_matrix[p, i] == 1 else np.zeros_like(sample[p])
                    for p in parents
                ]) 
            else:
                parent_values = np.zeros((sample_sizes[i], 0), dtype=float)

        
            f = f_list[node]

            # Evaluate structural signal and enforce expected vector shape.
            signal = np.asarray(f(parent_values), dtype=float).reshape(-1)
            # print(f"node {node} signal std {np.std(signal)}")
            if signal.size == 1:
                signal = np.full(sample_sizes[i], signal.item(), dtype=float)
            if signal.shape[0] != sample_sizes[i]:
                raise ValueError(f"f(node={node}) returned shape {signal.shape}, expected ({sample_sizes[i]},)")
            
            # Fetch context-specific node noise and validate shape.
            cur_noise = noise[node][i]
            if cur_noise.shape[0] != sample_sizes[i]:
                raise ValueError(f"noise for node {node} has shape {cur_noise.shape}, expected ({sample_sizes[i]},)")
            # print(f"node {node} noise std {np.std(cur_noise)}")
            # Final SEM equation: X_node = f(parents) + noise.
            sample[node] = signal + cur_noise
            # if len(parents)==0:
            #     print('investigate root', f'node {node} std {np.std(sample[node])}')
        samples.append(sample.copy())
    return samples, f_list, f_details


def find_root(samples, candidates, intervention_matrix):
    """
    Given context-wise samples and candidate nodes, pick root as:
    1) compute stability score per node (std of context-wise variances),
    2) keep nodes with score < 2 * (smallest score),
    3) among them, pick node with largest sum(intervention_matrix[node, :]).
    """
    nodes_stat_stable = {}

    for node in candidates:
        row_sum = np.sum(intervention_matrix[node, :])
        if row_sum == 0:
            continue

        weighted_vars = []
        for k in range(intervention_matrix.shape[1]):
            if intervention_matrix[node, k] == 1:
                weighted_vars.append(np.var(samples[k][node]) / row_sum)
        # print(f'node {node}, weighted_vars {weighted_vars}')
        if len(weighted_vars) < 2:
            continue
        

        nodes_stat_stable[node] = np.std(weighted_vars)
    if not nodes_stat_stable:
        raise ValueError("No valid candidate node found.")

    smallest = min(nodes_stat_stable.values())
    # print(max(nodes_stat_stable.values()))
    close_nodes = [n for n, v in nodes_stat_stable.items() if v < 5 * smallest]
    if len(close_nodes)>2:
        close_nodes = [n for n,_ in sorted(nodes_stat_stable.items(), key=lambda x: x[1])[:2]]
    # print(f"close nodes: {close_nodes}")

    # Pick by largest intervention-matrix row sum; tie-break by smaller stability score.
    root_idx = max(
        close_nodes,
        key=lambda n: (np.sum(intervention_matrix[n, :]), -nodes_stat_stable[n]),
    )
    return root_idx





def _reinit_linear_(module, generator):
    """Re-init every nn.Linear in `module` from `generator` (thread-local RNG),
    matching PyTorch's default kaiming_uniform_(a=sqrt(5)) init."""
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5), generator=generator)
            if m.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                bound = 1.0 / math.sqrt(fan_in) if fan_in > 0 else 0.0
                nn.init.uniform_(m.bias, -bound, bound, generator=generator)


class FFN_gate(nn.Module):
    """Simple feedforward neural network."""

    def __init__(self, in_dim, hidden_sizes=[32,32], activation='relu', generator=None):
        super().__init__() # super() returns a proxy to the parent class so one can call its methods

        # Build layers
        layers = []
        input_size = in_dim
        self.gate_logits = nn.Parameter(torch.ones(in_dim))  # sigmoid(0)=0.5

        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(input_size, hidden_size))
            if activation == 'tanh':
                layers.append(nn.Tanh())
            elif activation == 'relu':
                layers.append(nn.ReLU())
            input_size = hidden_size

        # Output layer (linear activation for regression)
        layers.append(nn.Linear(input_size, 1))

        self.network = nn.Sequential(*layers)

        if generator is not None:
            _reinit_linear_(self, generator)

    def forward(self, x):
        gates = torch.relu(torch.sigmoid(self.gate_logits)-0.5)
        x = 2*x*gates
        return self.network(x)


class FFN(nn.Module):
    """Simple feedforward neural network."""

    def __init__(self, in_dim, hidden_sizes=[32,32], activation='relu', generator=None):
        super().__init__() # super() returns a proxy to the parent class so one can call its methods

        # Build layers
        layers = []
        input_size = in_dim

        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(input_size, hidden_size))
            if activation == 'tanh':
                layers.append(nn.Tanh())
            elif activation == 'relu':
                layers.append(nn.ReLU())
            input_size = hidden_size

        # Output layer (linear activation for regression)
        layers.append(nn.Linear(input_size, 1))

        self.network = nn.Sequential(*layers)

        if generator is not None:
            _reinit_linear_(self, generator)

    def forward(self, x):
        return self.network(x)

def generate_data_next_node(node_to_fit, nodes_learned, samples, intervention_matrix, random_state = 42):
    
    num_contexts = intervention_matrix.shape[1]
    y_list = []
    x_list = []
    for i in range(num_contexts):
        x = []
        if intervention_matrix[node_to_fit,i]==1:
            y_list.append(samples[i][node_to_fit])
            for node in nodes_learned:
                # print('?',i,nodes_learned,intervention_matrix[node][i])
                if intervention_matrix[node][i]==1:
                    x.append(samples[i][node])
                else:
                    x.append(np.zeros_like(samples[i][node]))
            x_list.append(np.vstack(x))
    x_data = np.hstack(x_list).T
    y_data = np.hstack(y_list)
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, train_size=0.7, random_state=random_state, shuffle = True)
    if len(x_train.shape)==1:
        x_train = x_train.reshape(-1,1)
        x_test = x_test.reshape(-1,1)
    return torch.FloatTensor(x_train),torch.FloatTensor(y_train).unsqueeze(-1),torch.FloatTensor(x_test),torch.FloatTensor(y_test).unsqueeze(-1)


def fit_next_node(node_to_fit, nodes_learned, x_train, y_train,
                  x_val, y_val, intervention_matrix, epochs = 500,
                  verbose = True, gate = True, generator = None):
    criterion = nn.MSELoss()
    h = 32*round(np.sqrt(len(nodes_learned)))
    if gate:
        model = FFN_gate(in_dim = len(nodes_learned),hidden_sizes = [h,h], generator=generator)
        # print(f"===================== Training Model to fit {node_to_fit} with Gate ====================")
    else:
        model = FFN(in_dim = len(nodes_learned),hidden_sizes = [h,h], generator=generator)
        # print(f"===================== Training Model to fit {node_to_fit} without Gate ==================")
    optimizer = optim.Adam(model.parameters(), lr = 0.01)
    
    history = {'loss': [], 'val_loss': []}

    patience = 50
    wait = 0
    best_val = float("inf")
    best_state = None

    if verbose:
        print(f"[fit] node={node_to_fit} gate={gate} n_inputs={len(nodes_learned)} train_n={x_train.shape[0]} val_n={x_val.shape[0]} epochs={epochs}")

    epochs_run = 0
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        y_pred = model(x_train)

        loss = criterion(y_pred, y_train)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        model.eval()
        with torch.no_grad():
            y_val_pred = model(x_val)
            val_loss = criterion(y_val_pred, y_val)

        train_loss = loss.item()
        val_loss_item = val_loss.item()
        history['loss'].append(train_loss)
        history['val_loss'].append(val_loss_item)

        if val_loss_item < best_val - 1e-8:
            best_val = val_loss_item
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1

        epochs_run = epoch + 1
        if verbose and ((epoch + 1) % 50 == 0 or epoch == 0):
            print(f"[fit] node={node_to_fit} epoch={epoch+1}/{epochs} train_loss={train_loss:.6f} val_loss={val_loss_item:.6f} best_val={best_val:.6f} wait={wait}/{patience}")

        if wait >= patience:
            if verbose:
                print(f"[fit] node={node_to_fit} early_stop epoch={epoch+1} best_val={best_val:.6f}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    history['model'] = model

    if verbose:
        print(f"[fit] node={node_to_fit} done epochs_run={epochs_run} best_val={best_val:.6f}")
    
    return history

def node_stability_stat(samples, model, intervention_matrix, x_train, y_train,
                        x_val, y_val, node, parents, best_val_loss, verbose=True, gate_threshold = 1,
                        generator=None):
    parent_arr = np.asarray(parents)
    gate_mask = (model.gate_logits.detach().cpu().numpy().reshape(-1) > gate_threshold)
    if verbose:
        gate_vals = model.gate_logits.detach().cpu().numpy().round(4).tolist()
        print(f"[stability] node={node} threshold={gate_threshold} parents={parents} gate_logits={gate_vals}")
    if gate_mask.shape[0] != parent_arr.shape[0]:
        raise ValueError(f"gate size {gate_mask.shape[0]} != parents {parent_arr.shape[0]}")

    selected_parents = parent_arr[gate_mask].tolist()
    mask = np.flatnonzero(gate_mask).tolist()

    model_refined = None
    if len(selected_parents) > 0:
        if verbose:
            print(f"[stability] node={node} selected_from_gate={selected_parents} (k={len(selected_parents)}/{len(parents)})")

        history = fit_next_node(node, selected_parents, x_train[:, mask], y_train,
                                x_val[:, mask], y_val, intervention_matrix,
                                epochs=min(500*len(selected_parents),2000), verbose=verbose, gate=False,
                                generator=generator)
        if history['val_loss'][-1] < best_val_loss:
            model_refined = history["model"]
            if verbose:
                print(f"[stability] node={node} refined_model accepted: {history['val_loss'][-1]:.6f} < {best_val_loss:.6f}")
        else:
            model_refined = model
            selected_parents = parent_arr.tolist()
            if verbose:
                print(f"[stability] node={node} refined_model rejected: {history['val_loss'][-1]:.6f} >= {best_val_loss:.6f}; fallback_to_full={selected_parents}")

    def _predict(x):
        model_refined.eval()
        with torch.no_grad():
            y_hat = model_refined(torch.as_tensor(x, dtype=torch.float32)).cpu().numpy()
        return np.asarray(y_hat).reshape(-1)

    mse_list = []
    n_contexts = intervention_matrix.shape[1]
    for c in range(n_contexts):
        if intervention_matrix[node, c] != 1:
            continue

        y_true = np.asarray(samples[c][node]).reshape(-1)
        if len(selected_parents) > 0:
            cols = []
            for p in selected_parents:
                p_vals = np.asarray(samples[c][p]).reshape(-1)
                cols.append(p_vals if intervention_matrix[p, c] == 1 else np.zeros_like(p_vals))
            X = np.column_stack(cols)
            y_pred = _predict(X)
        else:
            y_pred = np.full_like(y_true, y_true.mean(), dtype=float)

        mse_list.append(float(np.mean((y_true - y_pred) ** 2)))

    if verbose and len(mse_list) > 0:
        arr = np.asarray(mse_list)
        print(f"[stability] node={node} parents_final={selected_parents} mse_mean={arr.mean():.6f} mse_std={arr.std():.6f} n_ctx={len(mse_list)}")

    return mse_list, selected_parents





def learn_graph(samples, intervention_matrix, verbose=False):
    n_nodes, n_contexts = intervention_matrix.shape
    graph = np.zeros((n_nodes, n_nodes), dtype=int)
    perm = []

    # optional: avoid mutating caller's samples
    samples = [{k: np.array(v, copy=True) for k, v in ctx.items()} for ctx in samples]

    root_idx = find_root(samples, np.arange(n_nodes), intervention_matrix)
    perm.append(root_idx)

    nodes_stat_stable_list = []

    while len(perm) < n_nodes:
        parents_list = {}
        nodes_stat_stable = {}
        models = {}
        feature_sets = {}
        stats_dict = {}

        for node in range(n_nodes):
            if node in perm:
                continue

            node_to_fit = node
            nodes_learned = list(perm)

            x_train, y_train, x_val, y_val = generate_data_next_node(
                node_to_fit, nodes_learned, samples, intervention_matrix
            )

            history = fit_next_node(
                node_to_fit, nodes_learned,
                x_train, y_train, x_val, y_val,
                intervention_matrix, epochs=min(500*len(nodes_learned),2000), verbose=verbose
            )
            model = history["model"]
            models[node] = model
            feature_sets[node] = nodes_learned

            stats, selected_parents = node_stability_stat(
                samples, model, intervention_matrix,
                x_train, y_train, x_val, y_val,
                node_to_fit, nodes_learned, history['val_loss'][-1],verbose=verbose
            )
            parents_list[node] = selected_parents

            row_sum = np.sum(intervention_matrix[node, :])
            if row_sum == 0:
                continue
            score_var = np.var(np.array(stats) / row_sum)
            score_mean_sq = np.mean(np.array(stats))**2
            score = score_var + 0.01 * score_mean_sq

            if verbose:
                print(f"[learn] candidate={node} selected_parents={selected_parents} score={score:.6f} var_term={score_var:.6f} mean_term={score_mean_sq:.6f} stats={np.round(stats, 6).tolist()}")

            nodes_stat_stable[node] = score
            stats_dict[node] = np.array(stats) / row_sum


        if not nodes_stat_stable:
            raise ValueError("No candidate node available; check intervention_matrix.")

        smallest = min(nodes_stat_stable.values())
        close_nodes = [n for n, v in nodes_stat_stable.items() if v < 5 * smallest]
        if len(close_nodes) > 2:
            close_nodes = [n for n, _ in sorted(nodes_stat_stable.items(), key=lambda x: x[1])[:2]]

        next_root_idx = max(
            close_nodes,
            key=lambda n: (np.sum(intervention_matrix[n, :]), -nodes_stat_stable[n]),
        )
        perm.append(next_root_idx)
        chosen_parents = parents_list[next_root_idx]
        if verbose:
            print(f"[learn] choose_next={next_root_idx} close_nodes={close_nodes} chosen_parents={chosen_parents} score={nodes_stat_stable[next_root_idx]:.6f}")
            print(f"[learn] current_order={perm}")
        nodes_stat_stable_list.append(stats_dict)
        if len(chosen_parents) > 0:
            graph[next_root_idx, chosen_parents] = 1

            # # residualize selected node using already-trained model (no refit)
            # model = models[next_root_idx]
            # trained_features = feature_sets[next_root_idx]  # full input set used by model
            # model.eval()

            # with torch.no_grad():
            #     for c in range(n_contexts):
            #         if intervention_matrix[next_root_idx, c] != 1:
            #             continue  # skip intervened target contexts

            #         cols = []
            #         for p in trained_features:
            #             p_vals = np.asarray(samples[c][p]).reshape(-1)
            #             if intervention_matrix[p, c] == 1:
            #                 cols.append(p_vals)
            #             else:
            #                 cols.append(np.zeros_like(p_vals))

            #         X_full = np.column_stack(cols).astype(np.float32)
            #         y_pred = model(torch.from_numpy(X_full)).cpu().numpy().reshape(-1)

            #         samples[c][next_root_idx] = (
            #             np.asarray(samples[c][next_root_idx]).reshape(-1) - y_pred
            #         )

    return perm, graph, nodes_stat_stable_list

def _candidate_job(node, perm, samples, intervention_matrix, verbose, base_seed=42):
    # per-job RNG: thread-local torch.Generator avoids cross-thread races on the
    # global default generator (which is what nn.Linear init normally uses).
    local_seed = base_seed * 100000 + len(perm) * 1000 + node
    gen = torch.Generator().manual_seed(local_seed)

    node_to_fit = node
    nodes_learned = list(perm)

    x_train, y_train, x_val, y_val = generate_data_next_node(
        node_to_fit, nodes_learned, samples, intervention_matrix
    )

    history = fit_next_node(
        node_to_fit, nodes_learned,
        x_train, y_train, x_val, y_val,
        intervention_matrix,
        epochs=min(500 * len(nodes_learned), 2000),
        verbose=verbose,
        generator=gen,
    )
    model = history["model"]

    stats, selected_parents = node_stability_stat(
        samples, model, intervention_matrix,
        x_train, y_train, x_val, y_val,
        node_to_fit, nodes_learned,
        history["val_loss"][-1],
        verbose=verbose,
        generator=gen,
    )

    row_sum = np.sum(intervention_matrix[node, :])
    if row_sum == 0:
        return None

    score_var = np.var(np.array(stats) / row_sum)
    score_mean_sq = np.mean(np.array(stats)) ** 2
    score = score_var + 0.01 * score_mean_sq

    return {
        "node": node,
        "parents": selected_parents,
        "score": score,
        "stats_norm": np.array(stats) / row_sum,
    }



def learn_graph_parallel(samples, intervention_matrix, verbose=False, max_workers=4, base_seed=42):
    n_nodes, n_contexts = intervention_matrix.shape
    graph = np.zeros((n_nodes, n_nodes), dtype=int)
    perm = []
    nodes_stat_stable_list = []

    samples = [{k: np.array(v, copy=True) for k, v in ctx.items()} for ctx in samples]

    root_idx = find_root(samples, np.arange(n_nodes), intervention_matrix)
    perm.append(root_idx)

    while len(perm) < n_nodes:
        candidates = [node for node in range(n_nodes) if node not in perm]

        results = []
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futs = [
                ex.submit(_candidate_job, node, perm, samples, intervention_matrix, verbose, base_seed)
                for node in candidates
            ]
            for fut in as_completed(futs):
                out = fut.result()
                if out is not None:
                    results.append(out)

        if not results:
            raise ValueError("No candidate node available; check intervention_matrix.")

        parents_list = {r["node"]: r["parents"] for r in results}
        nodes_stat_stable = {r["node"]: r["score"] for r in results}
        stats_dict = {r["node"]: r["stats_norm"] for r in results}

        smallest = min(nodes_stat_stable.values())
        close_nodes = [n for n, v in nodes_stat_stable.items() if v < 5 * smallest]
        if len(close_nodes) > 2:
            close_nodes = [n for n, _ in sorted(nodes_stat_stable.items(), key=lambda x: x[1])[:2]]

        next_root_idx = max(
            close_nodes,
            key=lambda n: (np.sum(intervention_matrix[n, :]), -nodes_stat_stable[n]),
        )
        perm.append(next_root_idx)
        nodes_stat_stable_list.append(stats_dict)

        chosen_parents = parents_list[next_root_idx]
        if len(chosen_parents) > 0:
            graph[next_root_idx, chosen_parents] = 1

        if verbose:
            print(f"[learn-par] choose_next={next_root_idx} close_nodes={close_nodes} parents={chosen_parents}")
            print(f"[learn-par] current_order={perm}")

    return perm, graph, nodes_stat_stable_list



if __name__ == "__main__":
    import time
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