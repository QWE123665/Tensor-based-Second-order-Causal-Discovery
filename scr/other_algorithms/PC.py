from causallearn.search.ConstraintBased.PC import pc 

def PC(X, alpha: float = 0.05):
    """
    PC algorithm for causal discovery.
    
    Parameters:
    - X: Data matrix (n_samples, n_features)
    - alpha: Significance level for conditional independence tests
    
    Returns:
    - Adjacency matrix representing the learned causal graph
    """
    
    # Run the PC algorithm using causallearn's implementation
    cg = pc(X, alpha=alpha,show_progress=False)
    
    # Extract the adjacency matrix from the learned graph
    return cg.G.graph, None