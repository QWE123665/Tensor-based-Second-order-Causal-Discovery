import numpy as np

### Functions to generate samples from certain Gaussian and non-Gaussian distributions


def gaussian_unit_rvs(size, rng):
    """Generate unit variance Gaussian random variables."""
    return rng.normal(size=size)


def uniform_unit_rvs(size, rng):
    """Generate unit variance uniform random variables."""
    lo, hi = -np.sqrt(3), np.sqrt(3)  # unit variance
    return rng.uniform(lo, hi, size=size)/ ((hi-lo)/np.sqrt(12))

def uniform_positive_rvs(size, rng):
    """Generate positive uniform random variables between 0.5 and 2."""
    return rng.uniform(0.1, 1, size=size)


def laplace_unit_rvs(size, rng):
    """Generate unit variance Laplace random variables."""
    return rng.laplace(0.0, 1 / np.sqrt(2), size=size)


def student_t_unit_rvs(df=5):
    """Generate unit variance Student's t random variables."""
    def _rvs(size, rng):
        z = rng.standard_t(df, size=size)
        assert np.all(np.isfinite(z))
        assert np.all(x is not None for x in z)
        return z / np.sqrt(df / (df - 2))
    return _rvs


DISTRIBUTIONS = {
    "gaussian": gaussian_unit_rvs,
    "uniform": uniform_unit_rvs,
    "laplace": laplace_unit_rvs,
    "t5": student_t_unit_rvs(df=5),
}

df_list = [i + 2 for i in np.logspace(0, 2, 20, dtype=int)]
for i in df_list:
    DISTRIBUTIONS[f't{i}'] = student_t_unit_rvs(df=i)


def resolve_rng(random_state=None):
    """Return a numpy Generator from a seed, Generator, or None."""
    if isinstance(random_state, np.random.Generator):
        return random_state
    return np.random.default_rng(random_state)


def default_edge_sampler(a1, b1, a2, b2):
    """Create a default edge sampler for uniform union distribution."""
    def uu(size, rng=None):
        rng = resolve_rng(rng)
        return uniform_union_two(rng, a1, b1, a2, b2, size)
    return uu


def uniform_union_two(rng, a1, b1, a2, b2, size):
    """
    Sample uniformly from [a1, b1] U [a2, b2].

    Args:
        rng: np.random.Generator
            Random number generator.
        a1, b1, a2, b2: float
            Interval endpoints.
        size: int or tuple
            Output shape.

    Returns:
        ndarray: Samples from the union of intervals.
    """
    if not (a1 < b1 and a2 < b2):
        raise ValueError("Bad interval endpoints.")
    len1, len2 = (b1 - a1), (b2 - a2)
    tot = len1 + len2
    if tot <= 0:
        raise ValueError("Total measure of union must be > 0.")

    N = int(np.prod(size))
    # Choose which interval each draw comes from
    take1 = rng.uniform(size=N) < (len1 / tot)

    out = np.empty(N, dtype=float)
    n1 = int(take1.sum())
    n2 = N - n1
    if n1:
        out[take1] = rng.uniform(a1, b1, size=n1)
    if n2:
        out[~take1] = rng.uniform(a2, b2, size=n2)
    return out.reshape(size)


def Lambda_sample_random(p, edge_prob, permute_nodes=True, permutation=None,
                         distribution=default_edge_sampler(-1, -0.4, 0.4, 1), random_state=None):
    """
    Sample a random DAG adjacency matrix.

    Args:
        p: int
            Number of nodes.
        edge_prob: float
            Probability of an edge.
        permute_nodes: bool
            Whether to permute nodes.
        permutation: ndarray or None
            Specific permutation.
        distribution: callable
            Distribution for edge weights.
        random_state: int or np.random.Generator or None
            Random state.

    Returns:
        adj_matrix: (p, p) ndarray
            Adjacency matrix.
        perm_inv: ndarray or None
            Inverse permutation if permuted.
    """
    rng = resolve_rng(random_state)

    # Generate random DAG adjacency matrix
    adj_matrix = distribution(size=(p, p), rng=rng)
    mask = rng.uniform(size=(p, p)) < edge_prob
    adj_matrix = adj_matrix * mask
    adj_matrix = np.tril(adj_matrix, k=-1)  # Make lower triangular to ensure acyclicity

    if permute_nodes:
        if permutation is not None:
            perm = permutation
        else:
            perm = rng.permutation(p)
        return adj_matrix[perm][:, perm], np.argsort(perm)
    return adj_matrix, np.arange(p)


def lsem_sample_perfect(Lambda, eps_var=None, zero_mask=[], sample_size=1,
                        eps_rvs=None, random_state=None, eps_var_distribution=uniform_positive_rvs,
                        return_intermediates=False, eps_mean=0):
    """
    Sample X from the LSEM: X = Lambda X + eps, i.e., (I - Lambda) X = eps.

    Args:
        Lambda: (p, p) array_like
            DAG adjacency matrix.
        eps_var: (p,) array_like or None
            Diagonal of Cov(eps). Must be positive.
        zero_mask: list
            Indices of nodes to hard intervene (zero out rows).
        sample_size: int
            Number of samples.
        eps_rvs: None or callable or list[callable]
            Noise samplers for non-Gaussian eps.
        random_state: int or np.random.Generator or None
            Random state.
        eps_var_distribution: callable
            Distribution for eps_var on intervened nodes.
        return_intermediates: Bool
            True means return X, E, Lambda instead of just X
        eps_mean: float or (p,) array_like
            Noise mean. Defaults to 1.0 so generated samples have nonzero mean.

    Returns:
        X: (sample_size, p) ndarray
            Samples.
    """
    n = sample_size
    rng = resolve_rng(random_state)

    Lambda_ = np.asarray(Lambda, dtype=float).copy()
    p = Lambda_.shape[0]
    if Lambda_.shape != (p, p):
        raise ValueError("Lambda must be square (p x p).")
    Lambda_[zero_mask==1,:] = 0.0

    if eps_var is not None:
        eps_var = np.asarray(eps_var, dtype=float).reshape(-1)
        if eps_var.shape[0] != p or np.any(eps_var <= 0):
            raise ValueError("eps_var must be length-p with all entries > 0.")
    else:
        eps_var = np.ones(p)

    eps_mean = np.asarray(eps_mean, dtype=float)
    if eps_mean.ndim == 0:
        eps_mean = np.full(p, float(eps_mean))
    else:
        eps_mean = eps_mean.reshape(-1)
        if eps_mean.shape[0] != p:
            raise ValueError("eps_mean must be scalar or length-p.")
    
    eps_var_new = eps_var.copy()  # Avoid modifying original
    if sum(zero_mask)>0:
        eps_var_new[np.where(zero_mask==1)[0]] = eps_var_distribution(size=np.sum(zero_mask), rng=rng)

    E = np.zeros((n, p))

    if eps_rvs is None:
        E = rng.multivariate_normal(
            mean=eps_mean,
            cov=np.diag(eps_var_new),
            size=n,
        )
    else:
        for i in range(p):
            samples = DISTRIBUTIONS[eps_rvs[i]](size=n, rng=rng)
            E[:, i] = samples * np.sqrt(eps_var_new[i]) + eps_mean[i]

    # Solve (I - Lambda) X = E -> X = (I - Lambda)^{-1} E
    A = np.eye(p) - Lambda_
    X = np.linalg.solve(A, E.T).T  # More stable than inv
    if return_intermediates:
        return X, E, Lambda_
    return X


def lsem_sample_soft(Lambda, eps_var=None, zero_mask=[], sample_size=1, eps_rvs=None, random_state=None,
                     lambda_distribution=default_edge_sampler(-1, -0.1, 0.1, 1), eps_var_distribution=uniform_positive_rvs,
                     eps_mean=0):
    """
    Sample X from the LSEM with soft interventions: modify Lambda for intervened nodes.

    Args:
        Lambda: (p, p) array_like
            DAG adjacency matrix.
        eps_var: (p,) array_like or None
            Diagonal of Cov(eps).
        zero_mask: list
            Indices of nodes to soft intervene.
        sample_size: int
            Number of samples.
        eps_rvs: None or callable or list[callable]
            Noise samplers.
        random_state: int or np.random.Generator or None
            Random state.
        lambda_distribution: callable
            Distribution for new Lambda entries.
        eps_var_distribution: callable
            Distribution for eps_var on intervened nodes.
        eps_mean: float or (p,) array_like
            Noise mean. Defaults to 1.0 so generated samples have nonzero mean.

    Returns:
        X: (sample_size, p) ndarray
            Samples.
        E: (sample_size, p) ndarray
            Noise samples used to generate X.
        Lambda_: (p, p) ndarray
            Modified adjacency matrix.
    """
    n = sample_size
    rng = resolve_rng(random_state)

    Lambda_ = np.asarray(Lambda, dtype=float).copy()
    p = Lambda_.shape[0]
    if Lambda_.shape != (p, p):
        raise ValueError("Lambda must be square (p x p).")

    Lambda_zero_support = (Lambda_ == 0)
    Lambda_[zero_mask==1, :] = lambda_distribution(size=(np.sum(zero_mask==1), p), rng=rng)
    Lambda_[Lambda_zero_support] = 0

    if eps_var is not None:
        eps_var = np.asarray(eps_var, dtype=float).reshape(-1)
        if eps_var.shape[0] != p or np.any(eps_var <= 0):
            raise ValueError("eps_var must be length-p with all entries > 0.")
    else:
        eps_var = np.ones(p)

    eps_mean = np.asarray(eps_mean, dtype=float)
    if eps_mean.ndim == 0:
        eps_mean = np.full(p, float(eps_mean))
    else:
        eps_mean = eps_mean.reshape(-1)
        if eps_mean.shape[0] != p:
            raise ValueError("eps_mean must be scalar or length-p.")
    
    eps_var = eps_var.copy()  # Avoid modifying original
    if sum(zero_mask)>0:
        eps_var[np.where(zero_mask==1)[0]] = eps_var_distribution(size=np.sum(zero_mask), rng=rng)

    E = np.zeros((n, p))

    if eps_rvs is None:
        E = rng.normal(size=(n, p)) * np.sqrt(eps_var)[None, :] + eps_mean[None, :]
    else:
        for i in range(p):
            samples = DISTRIBUTIONS[eps_rvs[i]](size=n, rng=rng)
            E[:, i] = samples * np.sqrt(eps_var[i]) + eps_mean[i]

    # Solve (I - Lambda) X = E -> X = (I - Lambda)^{-1} E
    A = np.eye(p) - Lambda_
    X = np.linalg.solve(A, E.T).T  # More stable than inv
    return X, E, Lambda_


def binary_code_array(n: int, observational=True) -> np.ndarray:
    """
    Return an (n, L) numpy array of 0/1 ints where L = floor(log2(n)) + 1,
    and rows are unique binary codes from 0...n-1 zero-padded to length L.
    If observational, prepend a column of ones if not all rows have 1 in first position.

    Args:
        n: int
            Number of rows.
        observational: bool
            Whether to add observational column.

    Returns:
        ndarray: Binary code array.
    """
    if n <= 0:
        raise ValueError("n must be a positive integer")

    L = n.bit_length()

    rows = [format(i, f'0{L}b') for i in range(n)]
    arr = np.array([[1 - int(ch) for ch in s] for s in rows], dtype=int)
    if observational and np.sum(arr[:, 0]) < n:
        arr = np.hstack((np.ones((n, 1), dtype=int), arr))
    return arr


def binary_code_single_intervene(n: int, observational=True) -> np.ndarray:
    """
    Return an (n, L) numpy array of 0/1 ints where L = n, and rows have exactly one 0 and n-1 1s.
    If observational, add an all-ones row.

    Args:
        n: int
            Number of interventions.
        observational: bool
            Whether to add observational row.

    Returns:
        ndarray: Binary code array for single interventions.
    """
    if n <= 0:
        raise ValueError("n must be a positive integer")

    arr = np.ones((n, n), dtype=int) - np.eye(n, dtype=int)
    if observational:
        arr = np.hstack((np.ones((n, 1), dtype=int), arr))
    return arr


def generate_LSEM_samples_perfect(n_nodes, edge_prob, sample_sizes, B, eps_var=None, eps_rvs=None,
                                  random_state=None, lambda_distribution=default_edge_sampler(-1, -0.1, 0.1, 1),
                                  eps_var_distribution=uniform_positive_rvs, permute_nodes=True,
                                  eps_mean=0):
    """
    Generate LSEM samples with perfect interventions.

    Args:
        n_nodes: int
            Number of nodes.
        edge_prob: float
            Edge probability.
        sample_sizes: list
            List of sample sizes.
        B: ndarray
            Intervention matrix.
        eps_var: (p,) array_like or None
            Noise variances.
        eps_rvs: None or list
            Noise distributions.
        random_state: int or np.random.Generator or None
            Random state.
        lambda_distribution: callable
            Distribution for Lambda.
        eps_var_distribution: callable
            Distribution for eps_var on intervened nodes.
        permute_nodes: bool
            Whether to permute nodes.
        eps_mean: float or (p,) array_like
            Noise mean passed to lsem_sample_perfect.

    Returns:
        Lambda: (p, p) ndarray
            Adjacency matrix.
        samples: list
            List of sample arrays.
        perm: ndarray
            Permutation applied to nodes.
    """
    rng = resolve_rng(random_state)
    Lambda, perm = Lambda_sample_random(n_nodes, edge_prob, permute_nodes=permute_nodes,
                                     random_state=rng, distribution=lambda_distribution)
    samples = []
    noises = []
    Lambdas = []

    for i, sample_size in enumerate(sample_sizes):
        X, E, Lambda_ = lsem_sample_perfect(
            Lambda,
            zero_mask=(1 - B[:, i]),
            eps_var=eps_var,
            eps_rvs=eps_rvs,
            random_state=rng,
            sample_size=sample_size,
            eps_var_distribution=eps_var_distribution,
            return_intermediates=True,
            eps_mean=eps_mean,
        )
        samples.append(X)
        noises.append(E)
        Lambdas.append(Lambda_)
    return Lambda, samples, perm, noises, Lambdas


def generate_LSEM_samples_soft(n_nodes, edge_prob, sample_sizes, B, eps_var=None, eps_rvs=None,
                               random_state=None, lambda_distribution=default_edge_sampler(-1, -0.1, 0.1, 1),
                               eps_var_distribution=uniform_positive_rvs, permute_nodes=True,
                               eps_mean=0):
    """
    Generate LSEM samples with soft interventions.

    Args:
        n_nodes: int
            Number of nodes.
        edge_prob: float
            Edge probability.
        sample_sizes: list
            List of sample sizes.
        B: ndarray
            Intervention matrix.
        eps_var: (p,) array_like or None
            Noise variances.
        eps_rvs: None or list
            Noise distributions.
        random_state: int or np.random.Generator or None
            Random state.
        lambda_distribution: callable
            Distribution for Lambda.
        eps_var_distribution: callable
            Distribution for eps_var on intervened nodes.
        permute_nodes: bool
            Whether to permute nodes.
        eps_mean: float or (p,) array_like
            Noise mean passed to lsem_sample_soft.

    Returns:
        Lambda_list: list
            List of modified adjacency matrices.
        samples: list
            List of sample arrays.
        perm: ndarray
            Permutation applied to nodes.
    """
    rng = resolve_rng(random_state)
    Lambda, perm = Lambda_sample_random(n_nodes, edge_prob, permute_nodes=permute_nodes,
                                     random_state=rng, distribution=lambda_distribution)
    
    samples = []
    Lambda_list = []
    noises = []
    for i, sample_size in enumerate(sample_sizes):
        X, E, Lambda_ = lsem_sample_soft(
            Lambda,
            zero_mask=(1 - B[:, i]),
            eps_var=eps_var,
            eps_rvs=eps_rvs,
            random_state=rng,
            sample_size=sample_size,
            lambda_distribution=lambda_distribution,
            eps_var_distribution=eps_var_distribution,
            eps_mean=eps_mean,
        )
        noises.append(E)
        samples.append(X)
        Lambda_list.append(Lambda_)
    return Lambda_list, samples, perm, noises


if __name__ == "__main__":
    pass
