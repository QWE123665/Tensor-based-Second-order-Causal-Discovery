from __future__ import annotations

import importlib
import itertools
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


THIS_FILE = Path(__file__).resolve()
if (THIS_FILE.parent / "generate_LSEM.py").exists():
    SCR_ROOT = THIS_FILE.parent
else:
    SCR_ROOT = THIS_FILE.parent / "scr"
PROJECT_ROOT = SCR_ROOT.parent
for import_root in (PROJECT_ROOT, SCR_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from scr import generate_LSEM as glsem


def linear_combo_gaussian_t5_unit(alpha: float):
    t5_sampler = glsem.student_t_unit_rvs(df=5)
    scale = np.sqrt(alpha**2 + (1 - alpha) ** 2)

    def _rvs(size, rng):
        gaussian = glsem.gaussian_unit_rvs(size=size, rng=rng)
        student_t = t5_sampler(size=size, rng=rng)
        if scale == 0:
            return np.zeros(size, dtype=float)
        return (alpha * gaussian + (1 - alpha) * student_t) / scale

    return _rvs


def register_linear_combo_distribution(alpha: float) -> str:
    name = f"lincomb_g_t5_alpha_{alpha:.4f}".replace(".", "p")
    glsem.DISTRIBUTIONS[name] = linear_combo_gaussian_t5_unit(alpha)
    return name


# Hyperparameters
@dataclass(frozen=True)
class Hyperparameters:
    run_name: str
    output_root: str | None
    methods: list[str]
    nodes: list[int]
    sample_sizes: list[int]
    edge_probs: list[float]
    n_trials: int
    seed: int
    observational_sample_size: int | None
    intervention_model: str
    binary_code_mode: str
    include_observational_env: bool
    custom_binary_code: Any
    noise_mode: str
    gaussian_ratios: list[float]
    tdistrs: list[str]
    noise_dists: list[str]
    mix_alphas: list[float]
    custom_eps_rvs: Any
    custom_eps_vars: Any


def build_metadata(
    hparams: Hyperparameters,
    method_specs: list["MethodSpec"],
    script_name: str,
    project_root: Path,
    scr_root: Path,
):
    return {
        "script": script_name,
        "created_unix": time.time(),
        "project_root": str(project_root),
        "scr_root": str(scr_root),
        "hyperparameters": {
            "run_name": hparams.run_name,
            "output_root": hparams.output_root,
            "methods": hparams.methods,
            "nodes": hparams.nodes,
            "sample_sizes": hparams.sample_sizes,
            "edge_probs": hparams.edge_probs,
            "n_trials": hparams.n_trials,
            "seed": hparams.seed,
            "observational_sample_size": hparams.observational_sample_size,
            "intervention_model": hparams.intervention_model,
            "binary_code_mode": hparams.binary_code_mode,
            "include_observational_env": hparams.include_observational_env,
            "custom_binary_code": hparams.custom_binary_code,
            "noise_mode": hparams.noise_mode,
            "gaussian_ratios": hparams.gaussian_ratios,
            "tdistrs": hparams.tdistrs,
            "noise_dists": hparams.noise_dists,
            "mix_alphas": hparams.mix_alphas,
            "custom_eps_rvs": hparams.custom_eps_rvs,
            "custom_eps_vars": hparams.custom_eps_vars,
        },
        "methods": [spec.name for spec in method_specs],
    }



# Methods
@dataclass(frozen=True)
class MethodSpec:
    name: str
    module_path: str
    function_name: str
    input_kind: str
    output_kind: str

    def load_callable(self):
        module = importlib.import_module(self.module_path)
        return getattr(module, self.function_name)
    

def load_method_callables(method_specs: list[MethodSpec]) -> tuple[dict[str, Any], dict[str, str]]:
    loaded: dict[str, Any] = {}
    load_errors: dict[str, str] = {}
    for spec in method_specs:
        try:
            loaded[spec.name] = spec.load_callable()
        except Exception as exc:
            load_errors[spec.name] = f"{type(exc).__name__}: {exc}"
    return loaded, load_errors



# Other helpers
def build_summary(results_df: pd.DataFrame) -> pd.DataFrame:
    if results_df.empty:
        return pd.DataFrame()

    group_columns = [
        "run_name",
        "method",
        "input_kind",
        "output_kind",
        "n_nodes",
        "sample_size_per_env",
        "observational_sample_size",
        "edge_prob",
        "binary_code_mode",
        "binary_code_source",
        "n_environments",
        "intervention_model",
        "noise_mode",
        "noise_label",
        "gaussian_ratio",
        "tdistr",
        "mix_alpha",
    ]

    summary = (
        results_df.groupby(group_columns, dropna=False)
        .agg(
            n_rows=("method", "size"),
            n_success=("status", lambda values: int(np.sum(values == "ok"))),
            runtime_sec_mean=("runtime_sec", "mean"),
            runtime_sec_std=("runtime_sec", "std"),
            frob_error_lambda_mean=("frob_error_lambda", "mean"),
            frob_error_lambda_std=("frob_error_lambda", "std"),
            f1_score_mean=("f1_score", "mean"),
            f1_score_std=("f1_score", "std"),
            tpr_mean=("tpr", "mean"),
            tpr_std=("tpr", "std"),
            shd_mean=("shd", "mean"),
            shd_std=("shd", "std"),
            wrong_parent_count_mean=("wrong_parent_count", "mean"),
            wrong_parent_count_std=("wrong_parent_count", "std"),
        )
        .reset_index()
    )
    summary["success_rate"] = summary["n_success"] / summary["n_rows"]
    return summary


def build_output_dir(hparams: Hyperparameters) -> Path:
    if hparams.output_root is None:
        root = Path.cwd() / "data" / hparams.run_name
    else:
        root = Path(hparams.output_root).expanduser() / hparams.run_name
    root.mkdir(parents=True, exist_ok=True)
    return root


def build_noise_settings(hparams: Hyperparameters, n_nodes: int) -> list[dict[str, Any]]:
    settings: list[dict[str, Any]] = []
    explicit_eps_rvs = None
    if hparams.custom_eps_rvs is not None:
        explicit_eps_rvs = [str(value) for value in hparams.custom_eps_rvs]
        if len(explicit_eps_rvs) != n_nodes:
            raise ValueError(f"CUSTOM_EPS_RVS must have length {n_nodes}, got {len(explicit_eps_rvs)}.")

    if hparams.noise_mode == "explicit":
        if explicit_eps_rvs is None:
            raise ValueError("NOISE_MODE='explicit' requires CUSTOM_EPS_RVS.")
        settings.append(
            {
                "noise_mode": "explicit",
                "noise_label": "explicit_eps_rvs",
                "eps_rvs": explicit_eps_rvs,
                "gaussian_ratio": np.nan,
                "tdistr": None,
                "mix_alpha": np.nan,
            }
        )
        return settings

    if hparams.noise_mode == "homogeneous":
        for dist in hparams.noise_dists:
            settings.append(
                {
                    "noise_mode": "homogeneous",
                    "noise_label": dist,
                    "eps_rvs": [dist] * n_nodes,
                    "gaussian_ratio": np.nan,
                    "tdistr": None,
                    "mix_alpha": np.nan,
                }
            )
        return settings

    if hparams.noise_mode == "linear_combo_g_t5":
        for alpha in hparams.mix_alphas:
            dist_name = register_linear_combo_distribution(alpha)
            settings.append(
                {
                    "noise_mode": "linear_combo_g_t5",
                    "noise_label": dist_name,
                    "eps_rvs": [dist_name] * n_nodes,
                    "gaussian_ratio": np.nan,
                    "tdistr": "t5",
                    "mix_alpha": alpha,
                }
            )
        return settings

    for gaussian_ratio, tdistr in itertools.product(hparams.gaussian_ratios, hparams.tdistrs):
        n_gaussian = int(n_nodes * gaussian_ratio)
        eps_rvs = ["gaussian"] * n_gaussian + [tdistr] * (n_nodes - n_gaussian)
        settings.append(
            {
                "noise_mode": "gaussian_ratio",
                "noise_label": f"gaussian_ratio_{gaussian_ratio:g}_{tdistr}",
                "eps_rvs": eps_rvs,
                "gaussian_ratio": gaussian_ratio,
                "tdistr": tdistr,
                "mix_alpha": np.nan,
            }
        )
    return settings


def build_binary_code(hparams: Hyperparameters, n_nodes: int) -> tuple[np.ndarray, str, str]:
    if hparams.binary_code_mode == "custom":
        if hparams.custom_binary_code is None:
            raise ValueError("BINARY_CODE_MODE='custom' requires CUSTOM_BINARY_CODE.")
        if not isinstance(hparams.custom_binary_code, np.ndarray):
            raise TypeError("CUSTOM_BINARY_CODE must be a NumPy array.")
        B = np.asarray(hparams.custom_binary_code, dtype=int)
        if B.ndim != 2:
            raise ValueError("CUSTOM_BINARY_CODE must be a 2D array.")
        if B.shape[0] != n_nodes and B.shape[1] == n_nodes:
            B = B.T
        if B.shape[0] != n_nodes:
            raise ValueError(
                f"CUSTOM_BINARY_CODE must have {n_nodes} rows (or columns that transpose to that). Got {B.shape}."
            )
        if not np.isin(B, [0, 1]).all():
            raise ValueError("CUSTOM_BINARY_CODE must contain only 0/1 values.")
        return B, "custom", "custom"

    observational = hparams.include_observational_env
    if hparams.binary_code_mode == "single_intervene":
        B = glsem.binary_code_single_intervene(n_nodes, observational=observational)
    elif hparams.binary_code_mode == "binary_array":
        B = glsem.binary_code_array(n_nodes, observational=observational)
    else:
        raise ValueError(f"Unknown BINARY_CODE_MODE: {hparams.binary_code_mode}")
    return B, hparams.binary_code_mode, hparams.binary_code_mode


def order_to_json(order: Any) -> str | None:
    if order is None:
        return None
    return json.dumps(np.asarray(order, dtype=int).tolist())
