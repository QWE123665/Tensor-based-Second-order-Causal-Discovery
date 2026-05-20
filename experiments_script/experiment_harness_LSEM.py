#!/usr/bin/env python3
from __future__ import annotations

import itertools
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


THIS_FILE = Path(__file__).resolve()
if (THIS_FILE.parents[1] / "generate_LSEM.py").exists():
    SCR_ROOT = THIS_FILE.parents[1]
else:
    SCR_ROOT = THIS_FILE.parents[1] / "scr"
PROJECT_ROOT = SCR_ROOT.parent
for import_root in (PROJECT_ROOT, SCR_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from scr import generate_LSEM as glsem
from scr import metrics
from harness_helpers import (
    Hyperparameters,
    MethodSpec,
    build_binary_code,
    build_metadata,
    build_noise_settings,
    build_output_dir,
    build_summary,
    load_method_callables,
    order_to_json,
)


# Edit these hyperparameters directly
RUN_NAME = time.strftime("experiment_%Y%m%d_%H%M%S")

OUTPUT_ROOT = None  # None -> <cwd>/data/<RUN_NAME>
METHODS = [
    "GES",
    "PC",
    "GIES",
    "IGSP",
    "SortRegress",
    "lingam_from_interventions_ica",
    "lingam_from_interventions_direct",
    "notears_linear",
    "TSCD",
]

# Sweep over Cartesian product of NODES, SAMPLE_SIZES, EDGE_PROBS
NODES = [10]
SAMPLE_SIZES = [int(x) for x in list(np.logspace(np.log10(100), np.log10(10000), 20, dtype=int)) ]
EDGE_PROBS = [0.6] # TODO: If edge prob <1, account for topological orders allowed

N_TRIALS = 30
SEED = 42
OBSERVATIONAL_SAMPLE_SIZE = None  # None -> sample_size_per_env * n_environments
INTERVENTION_MODEL = "perfect"  

# Binary code setup.
BINARY_CODE_MODE = "binary_array"  # "binary_array", "single_intervene", or "custom"
INCLUDE_OBSERVATIONAL_ENV = True
CUSTOM_BINARY_CODE = None  # None or np.ndarray with shape (n_nodes, n_environments)

# Noise setup.
NOISE_MODE = "gaussian_ratio"  
    # "gaussian_ratio" (some nodes gaussian, some nodes TDISTR), 
    # "homogeneous", 
    # "linear_combo_g_t5", or 
    # "explicit" (set CUSTOM hyperparams)
GAUSSIAN_RATIOS = [0, 0.5, 1.0]
TDISTRS = ["t5"]
NOISE_DISTS = ["gaussian"]
MIX_ALPHAS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
CUSTOM_EPS_RVS = None  # None or list[str] of length n_nodes
CUSTOM_EPS_VARS = None  # None, scalar, or array-like of length n_nodes


HYPERPARAMETERS = Hyperparameters(
    run_name=RUN_NAME,
    output_root=OUTPUT_ROOT,
    methods=METHODS,
    nodes=NODES,
    sample_sizes=SAMPLE_SIZES,
    edge_probs=EDGE_PROBS,
    n_trials=N_TRIALS,
    seed=SEED,
    observational_sample_size=OBSERVATIONAL_SAMPLE_SIZE,
    intervention_model=INTERVENTION_MODEL,
    binary_code_mode=BINARY_CODE_MODE,
    include_observational_env=INCLUDE_OBSERVATIONAL_ENV,
    custom_binary_code=CUSTOM_BINARY_CODE,
    noise_mode=NOISE_MODE,
    gaussian_ratios=GAUSSIAN_RATIOS,
    tdistrs=TDISTRS,
    noise_dists=NOISE_DISTS,
    mix_alphas=MIX_ALPHAS,
    custom_eps_rvs=CUSTOM_EPS_RVS,
    custom_eps_vars=CUSTOM_EPS_VARS,
)



METHOD_REGISTRY: dict[str, MethodSpec] = {
    "GES": MethodSpec("GES", "scr.other_algorithms.GES", "GES", "observational", "graph_causallearn"),
    "PC": MethodSpec("PC", "scr.other_algorithms.PC", "PC", "observational", "graph_causallearn"),
    "GIES": MethodSpec("GIES", "scr.other_algorithms.GIES", "GIES", "interventional", "graph_gies"),
    "IGSP": MethodSpec("IGSP", "scr.other_algorithms.IGSP", "IGSP", "interventional", "lambda"),
    "SortRegress": MethodSpec("SortRegress", "scr.other_algorithms.sort_regress", "SortRegress", "observational", "lambda"),
    "lingam_from_interventions_ica": MethodSpec(
        "lingam_from_interventions_ica",
        "scr.other_algorithms.LinGAM",
        "lingam_from_interventions_ica",
        "interventional",
        "lambda",
    ),
    "lingam_from_interventions_direct": MethodSpec(
        "lingam_from_interventions_direct",
        "scr.other_algorithms.LinGAM",
        "lingam_from_interventions_direct",
        "interventional",
        "lambda",
    ),
    "notears_linear": MethodSpec("notears_linear", "scr.other_algorithms.Notears_linear", "notears_linear", "observational", "lambda"),
    "TSCD": MethodSpec(
        "TSCD",
        "scr.TSCD",
        "TSCD",
        "interventional",
        "lambda",
    ),
}

# Helpers for compute_metrics
# ====================================
def directed_support_from_causallearn_graph(graph: np.ndarray) -> np.ndarray:
    graph = np.asarray(graph)
    n_nodes = graph.shape[0]
    support = np.zeros((n_nodes, n_nodes), dtype=int)
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            a_ij = graph[i, j]
            a_ji = graph[j, i]
            if a_ij == 1 and a_ji == -1:
                support[i, j] = 1
            elif a_ij == -1 and a_ji == 1:
                support[j, i] = 1
            elif a_ij != 0 or a_ji != 0:
                support[i, j] = 1
                support[j, i] = 1
    return support


def directed_support_from_gies_graph(graph: np.ndarray) -> np.ndarray:
    graph = np.asarray(graph)
    n_nodes = graph.shape[0]
    support = np.zeros((n_nodes, n_nodes), dtype=int)
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            a_ij = graph[i, j]
            a_ji = graph[j, i]
            if a_ij == 1 and a_ji == 0:
                support[i, j] = 1
            elif a_ij == 0 and a_ji == 1:
                support[j, i] = 1
            elif a_ij != 0 or a_ji != 0:
                support[i, j] = 1
                support[j, i] = 1
    return support


def compute_metrics(spec: MethodSpec, estimate: np.ndarray, lambda_true: np.ndarray, graph_true: np.ndarray) -> dict[str, float]:
    result = {
        "frob_error_lambda": np.nan,
        "f1_score": np.nan,
        "tpr": np.nan,
        "shd": np.nan,
    }

    if spec.output_kind == "lambda":
        lambda_est = np.asarray(estimate, dtype=float)
        graph_est = metrics.binarize_adjacency(lambda_est)
        result["frob_error_lambda"] = metrics.relative_frob_score(lambda_est, lambda_true)
        result["f1_score"] = metrics.safe_f1_score(lambda_est, lambda_true)
        result["tpr"] = float(metrics.tpr_p(graph_true, graph_est))
        result["shd"] = float(metrics.shd(graph_true, graph_est))
        return result

    if spec.output_kind == "graph_causallearn":
        support_est = directed_support_from_causallearn_graph(estimate)
        result["f1_score"] = metrics.safe_f1_score(support_est, graph_true)
        result["tpr"] = float(metrics.tpr_p(graph_true, estimate))
        result["shd"] = float(metrics.shd_binary(graph_true, estimate))
        return result

    if spec.output_kind == "graph_gies":
        support_est = directed_support_from_gies_graph(estimate)
        result["f1_score"] = metrics.safe_f1_score(support_est, graph_true)
        result["tpr"] = float(metrics.tpr_p(graph_true, estimate))
        result["shd"] = float(metrics.shd_binary_2(graph_true, estimate))
        return result

    raise ValueError(f"Unknown output_kind: {spec.output_kind}")

# ====================================


def sample_environment_data(
    lambda_true: np.ndarray,
    B: np.ndarray,
    sample_size_per_env: int,
    observational_sample_size: int,
    eps_rvs: list[str],
    eps_var: np.ndarray | None,
    intervention_model: str,
    rng: np.random.Generator,
) -> tuple[list[np.ndarray], np.ndarray]:
    X_list: list[np.ndarray] = []
    env_sampler = glsem.lsem_sample_perfect 

    for env_idx in range(B.shape[1]):
        zero_mask = (B[:, env_idx] == 0).astype(int)
        if intervention_model == "perfect":
            X = env_sampler(
                lambda_true,
                eps_var=eps_var,
                zero_mask=zero_mask,
                sample_size=sample_size_per_env,
                eps_rvs=eps_rvs,
                random_state=rng,
            )
        else:
            X, _, _ = env_sampler(
                lambda_true,
                eps_var=eps_var,
                zero_mask=zero_mask,
                sample_size=sample_size_per_env,
                eps_rvs=eps_rvs,
                random_state=rng,
            )
        X_list.append(X)

    Y = glsem.lsem_sample_perfect(
        lambda_true,
        eps_var=eps_var,
        zero_mask=np.zeros(lambda_true.shape[0], dtype=int),
        sample_size=observational_sample_size,
        eps_rvs=eps_rvs,
        random_state=rng,
    )
    return X_list, Y



def run_single_method(method_callable, spec: MethodSpec, X_list: list[np.ndarray], Y: np.ndarray, B: np.ndarray):
    start = time.perf_counter()
    try:
        output = method_callable(X_list, B) if spec.input_kind == "interventional" else method_callable(Y)
        runtime_sec = time.perf_counter() - start
        estimate = output[0] if isinstance(output, tuple) else output
        estimated_order = output[1] if isinstance(output, tuple) and len(output) > 1 else None
        return "ok", estimate, estimated_order, runtime_sec, ""
    
    except Exception as exc:
        runtime_sec = time.perf_counter() - start
        return "error", None, None, runtime_sec, f"{type(exc).__name__}: {exc}"



def selected_method_specs(method_names: list[str]) -> list[MethodSpec]:
    missing = [name for name in method_names if name not in METHOD_REGISTRY]
    if missing:
        available = ", ".join(sorted(METHOD_REGISTRY))
        raise ValueError(f"Unknown methods: {missing}. Available methods: {available}")
    return [METHOD_REGISTRY[name] for name in method_names]



def run_experiments(hparams: Hyperparameters) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """
    Run the full experiment sweep and return: raw results, summary results, and metadata.
    """
    experiment_start = time.perf_counter()

    results: list[dict[str, Any]] = []
    method_specs = selected_method_specs(hparams.methods)
    loaded_methods, load_errors = load_method_callables(method_specs)

    for n_nodes, edge_prob in itertools.product(hparams.nodes, hparams.edge_probs):
        print("n_nodes: ", n_nodes)
        print("edge_prob: ", edge_prob)
        print("---==========---")

        B, binary_code_mode, binary_code_source = build_binary_code(hparams, n_nodes)
        noise_settings = build_noise_settings(hparams, n_nodes)
        n_environments = int(B.shape[1])

        for trial in range(hparams.n_trials):
            print("trial: ", trial)
            print("==========")

            trial_rng = np.random.default_rng([hparams.seed, trial, n_nodes])
            eps_var = trial_rng.uniform(0.1, 1, size=n_nodes)
            lambda_true, true_causal_order = glsem.Lambda_sample_random(
                p=n_nodes,
                edge_prob=edge_prob,
                permute_nodes=True,
                random_state=trial_rng,
            )
            graph_true = metrics.binarize_adjacency(lambda_true)
            true_dag = metrics.adjacency_to_parent_dict(graph_true)

            for sample_size_per_env, noise_setting in itertools.product(hparams.sample_sizes, noise_settings):
                print("sample_size_per_env: ", sample_size_per_env)

                observational_sample_size = (
                    hparams.observational_sample_size
                    if hparams.observational_sample_size is not None
                    else sample_size_per_env * n_environments
                )

                X_list, Y = sample_environment_data(
                    lambda_true=lambda_true,
                    B=B,
                    sample_size_per_env=sample_size_per_env,
                    observational_sample_size=observational_sample_size,
                    eps_rvs=noise_setting["eps_rvs"],
                    eps_var=eps_var,
                    intervention_model=hparams.intervention_model,
                    rng=trial_rng,
                )

                for spec in method_specs:
                    print("method: ", spec.name)

                    if spec.name in load_errors:
                        status = "error"
                        estimate = None
                        estimated_causal_order = None
                        runtime_sec = np.nan
                        error_message = load_errors[spec.name]
                    else:
                        status, estimate, estimated_causal_order, runtime_sec, error_message = run_single_method(
                            method_callable=loaded_methods[spec.name],
                            spec=spec,
                            X_list=X_list,
                            Y=Y,
                            B=B,
                        )

                    print("Post-Run")
                    print("status: ", status)
                    print("runtime_sec: ", runtime_sec)

                    row = {
                        "run_name": hparams.run_name,
                        "trial": trial,
                        "seed": hparams.seed,
                        "method": spec.name,
                        "input_kind": spec.input_kind,
                        "output_kind": spec.output_kind,
                        "status": status,
                        "error_message": error_message,
                        "runtime_sec": runtime_sec,
                        "n_nodes": n_nodes,
                        "sample_size_per_env": sample_size_per_env,
                        "observational_sample_size": observational_sample_size,
                        "edge_prob": edge_prob,
                        "binary_code_mode": binary_code_mode,
                        "binary_code_source": binary_code_source,
                        "n_environments": n_environments,
                        "binary_code_shape": f"{B.shape[0]}x{B.shape[1]}",
                        "intervention_model": hparams.intervention_model,
                        "noise_mode": noise_setting["noise_mode"],
                        "noise_label": noise_setting["noise_label"],
                        "gaussian_ratio": noise_setting["gaussian_ratio"],
                        "tdistr": noise_setting["tdistr"],
                        "mix_alpha": noise_setting["mix_alpha"],
                        "eps_rvs_json": json.dumps(noise_setting["eps_rvs"]),
                        "eps_var_json": None if eps_var is None else json.dumps(eps_var.astype(float).tolist()),
                        "true_causal_order_json": order_to_json(true_causal_order),
                        "estimated_causal_order_json": order_to_json(estimated_causal_order),
                        "wrong_parent_count": metrics.count_wrong_parents(estimated_causal_order, true_dag),
                    }

                    if status == "ok" and estimate is not None:
                        row.update(
                            compute_metrics(
                                spec=spec,
                                estimate=np.asarray(estimate),
                                lambda_true=lambda_true,
                                graph_true=graph_true,
                            )
                        )
                    else:
                        row.update(
                            {
                                "frob_error_lambda": np.nan,
                                "f1_score": np.nan,
                                "tpr": np.nan,
                                "shd": np.nan,
                            }
                        )

                    results.append(row)

    results_df = pd.DataFrame(results)
    summary_df = build_summary(results_df)
    metadata = build_metadata(
        hparams,
        method_specs,
        script_name=THIS_FILE.name,
        project_root=PROJECT_ROOT,
        scr_root=SCR_ROOT,
    )

    experiment_sec = time.perf_counter() - experiment_start
    print("Total experiment time: ", experiment_sec)

    return results_df, summary_df, metadata


def main() -> None:
    print("Running experiment main(). \n")

    out_dir = build_output_dir(HYPERPARAMETERS)
    results_df, summary_df, metadata = run_experiments(HYPERPARAMETERS)

    raw_csv = out_dir / "raw_results.csv"
    summary_csv = out_dir / "summary_results.csv"
    metadata_json = out_dir / "metadata.json"

    results_df.to_csv(raw_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)
    metadata_json.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Saved raw results to {raw_csv}")
    print(f"Saved summary results to {summary_csv}")
    print(f"Saved metadata to {metadata_json}")


if __name__ == "__main__":
    main()
