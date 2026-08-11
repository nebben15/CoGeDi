from __future__ import annotations

import csv
import datetime
import io
import json
import os
import queue
import re
import threading
from contextlib import ExitStack, nullcontext, redirect_stderr, redirect_stdout
from copy import deepcopy
from types import SimpleNamespace
from typing import Any, Dict, Iterable, Optional, Tuple

import optuna
import yaml
from tqdm import tqdm

from cogedi.registry import register_all
import cogedi.orch.train as train
import cogedi.orch.eval as eval


def _to_namespace(obj):
    if isinstance(obj, dict):
        return SimpleNamespace(**{k: _to_namespace(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_to_namespace(v) for v in obj]
    return obj


def _load_config_raw(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        cfg_dict = yaml.safe_load(f)
    if cfg_dict is None:
        raise ValueError("Empty config file")
    if not isinstance(cfg_dict, dict):
        raise ValueError("Config must be a dict at the top level")
    return cfg_dict


def _resolve_env_select(obj: Any, env_name: str) -> Any:
    if isinstance(obj, dict):
        keys = set(obj.keys())
        selector_keys = {"local", "slurm", "default"}
        if keys and keys.issubset(selector_keys):
            if env_name in obj:
                return _resolve_env_select(obj[env_name], env_name)
            if "default" in obj:
                return _resolve_env_select(obj["default"], env_name)
        return {k: _resolve_env_select(v, env_name) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_resolve_env_select(v, env_name) for v in obj]
    if isinstance(obj, str):
        return os.path.expandvars(obj)
    return obj


def _get_attr(obj: Dict[str, Any], path: str, default: Any) -> Any:
    cur: Any = obj
    for key in path.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _set_attr(obj: Dict[str, Any], path: str, value: Any) -> None:
    keys = path.split(".")
    cur = obj
    for key in keys[:-1]:
        if key not in cur or not isinstance(cur[key], dict):
            cur[key] = {}
        cur = cur[key]
    cur[keys[-1]] = value


def _spec_to_dict(spec: Any, *, label: str) -> Dict[str, Any]:
    if isinstance(spec, dict):
        return spec
    spec_dict = getattr(spec, "__dict__", None)
    if isinstance(spec_dict, dict):
        return dict(spec_dict)
    raise ValueError(f"Each {label} entry must be a dict")


def _resolve_eval_output_dir(cfg_dict: Dict[str, Any]) -> Optional[str]:
    paths_cfg = cfg_dict.get("paths", {}) if isinstance(cfg_dict.get("paths"), dict) else {}
    eval_base = paths_cfg.get("eval")

    # If eval_base is a selector structure (dict or SimpleNamespace), resolve
    # it using the run env in the provided cfg_dict. This makes the resolver
    # robust if env-selection wasn't applied earlier.
    selector_keys = {"local", "slurm", "default"}
    if eval_base is not None:
        if isinstance(eval_base, dict):
            obj = eval_base
        elif isinstance(eval_base, SimpleNamespace) or hasattr(eval_base, "__dict__"):
            obj = getattr(eval_base, "__dict__", {})
        else:
            obj = None

        if obj and set(obj.keys()).issubset(selector_keys):
            run_cfg = cfg_dict.get("run", {}) if isinstance(cfg_dict.get("run"), dict) else {}
            env = run_cfg.get("env") or run_cfg.get("environment") or "local"
            env = str(env).lower()
            if env in obj:
                eval_base = obj[env]
            elif "default" in obj:
                eval_base = obj["default"]
            else:
                eval_base = None

    exp_name = _get_attr(cfg_dict, "run.experiment_name", None)
    if eval_base and exp_name:
        return os.path.join(eval_base, exp_name)
    return eval_base


def _parse_eval_report(report_path: str) -> Dict[str, float]:
    if not os.path.exists(report_path):
        return {}

    metrics: Dict[str, float] = {}
    section = None
    current_mod = None

    def _as_float(val: str) -> Optional[float]:
        try:
            return float(val)
        except Exception:
            return None

    with open(report_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("==="):
                section = line.strip("=").strip().lower().replace(" ", "_")
                current_mod = None
                continue
            if line.startswith("[Modality:"):
                match = re.search(r"\[Modality:\s*(.+)\]", line)
                current_mod = match.group(1).strip() if match else None
                continue

            if ":" not in line:
                continue
            key, val = [s.strip() for s in line.split(":", 1)]
            val_f = _as_float(val)
            if val_f is None:
                continue

            metric_key = key
            if section:
                metric_key = f"{section}.{metric_key}"
            if current_mod:
                metric_key = f"{metric_key}/{current_mod}"
            metrics[metric_key] = val_f

    return metrics


def _filter_metrics(metrics: Dict[str, float], include: Iterable[str], exclude: Iterable[str]) -> Dict[str, float]:
    include = [s for s in include if s]
    exclude = [s for s in exclude if s]

    def _keep(k: str) -> bool:
        if include and not any(k.startswith(p) or p in k for p in include):
            return False
        if exclude and any(k.startswith(p) or p in k for p in exclude):
            return False
        return True

    return {k: v for k, v in metrics.items() if _keep(k)}


def _suggest_value(trial: optuna.Trial, spec: Dict[str, Any]) -> Any:
    name = str(spec.get("name"))
    kind = str(spec.get("type", "float")).lower()

    if kind == "float":
        low = float(spec["low"])
        high = float(spec["high"])
        step = spec.get("step", None)
        log = bool(spec.get("log", False))
        if step is not None:
            return trial.suggest_float(name, low, high, step=float(step), log=log)
        return trial.suggest_float(name, low, high, log=log)

    if kind == "int":
        low = int(spec["low"])
        high = int(spec["high"])
        step = int(spec.get("step", 1))
        log = bool(spec.get("log", False))
        return trial.suggest_int(name, low, high, step=step, log=log)

    if kind == "categorical":
        choices = spec.get("choices")
        if not isinstance(choices, list):
            raise ValueError(f"categorical param '{name}' must provide a list of choices")
        return trial.suggest_categorical(name, choices)

    if kind == "fixed":
        return spec.get("value")

    raise ValueError(f"Unknown param type: {kind}")


def _ablation_values(spec: Dict[str, Any]) -> list[Any]:
    if "values" in spec:
        values = spec.get("values")
        if not isinstance(values, list):
            raise ValueError("ablation values must be a list")
        return values

    kind = str(spec.get("type", "float")).lower()
    if kind == "categorical":
        choices = spec.get("choices")
        if not isinstance(choices, list):
            raise ValueError("categorical params require a list of choices")
        return choices

    if kind == "int":
        low = int(spec["low"])
        high = int(spec["high"])
        step = int(spec.get("step", 1))
        return list(range(low, high + 1, step))

    if kind == "float":
        if "step" not in spec:
            raise ValueError("float params in ablation mode require 'values' or a 'step'")
        low = float(spec["low"])
        high = float(spec["high"])
        step = float(spec["step"])
        values: list[float] = []
        cur = low
        while cur <= high + 1e-12:
            values.append(float(cur))
            cur += step
        return values

    if kind == "fixed":
        return [spec.get("value")]

    raise ValueError(f"Unknown param type: {kind}")


def _build_grid_search_space(params: list[Dict[str, Any]]) -> Dict[str, list[Any]]:
    space: Dict[str, list[Any]] = {}
    for spec in params:
        name = str(spec.get("name"))
        values = _ablation_values(spec)
        if not values:
            raise ValueError(f"grid sampler requires non-empty values for '{name}'")
        space[name] = values
    return space


def _build_sampler(cfg: Dict[str, Any], params: list[Dict[str, Any]]) -> optuna.samplers.BaseSampler:
    sampler = str(_get_attr(cfg, "hpo.optuna.sampler", "tpe")).lower()
    seed = _get_attr(cfg, "hpo.optuna.seed", None)
    if sampler == "random":
        return optuna.samplers.RandomSampler(seed=seed)
    if sampler == "tpe":
        return optuna.samplers.TPESampler(seed=seed)
    if sampler == "grid":
        grid = _get_attr(cfg, "hpo.optuna.grid_search_space", None)
        if grid is None:
            grid = _build_grid_search_space(params)
        if not isinstance(grid, dict) or not grid:
            raise ValueError("grid_search_space must be a non-empty dict of param -> list")
        for key, values in grid.items():
            if not isinstance(values, list) or not values:
                raise ValueError(f"grid_search_space for '{key}' must be a non-empty list")
        return optuna.samplers.GridSampler(search_space=grid)
    raise ValueError("hpo.optuna.sampler must be one of: tpe, random, grid")


def _build_pruner(cfg: Dict[str, Any]) -> optuna.pruners.BasePruner:
    pruner = str(_get_attr(cfg, "hpo.optuna.pruner", "median")).lower()
    if pruner == "none":
        return optuna.pruners.NopPruner()
    if pruner == "median":
        return optuna.pruners.MedianPruner()
    raise ValueError("hpo.optuna.pruner must be one of: median, none")


def _write_csv(path: str, rows: list[Dict[str, Any]]) -> None:
    if not rows:
        return
    keys = sorted({k for row in rows for k in row.keys()})
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _find_latest_summary(out_dir: str) -> Optional[str]:
    if not os.path.isdir(out_dir):
        return None
    candidates = []
    for name in os.listdir(out_dir):
        if not name.startswith("hpo_summary_") or not name.endswith(".csv"):
            continue
        candidates.append(os.path.join(out_dir, name))
    if not candidates:
        return None
    candidates.sort()
    return candidates[-1]


def _load_csv(path: str) -> list[Dict[str, Any]]:
    if not path or not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return [dict(row) for row in reader]


def _resolve_mlflow_uri(path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    s = str(path)
    # If the user provided a full URI (contains scheme), trust it as-is.
    if "://" in s:
        # Normalize sqlite URIs: allow user to provide sqlite:///path or
        # sqlite:////absolute/path; ensure absolute paths have four slashes
        if s.startswith("sqlite:///"):
            rest = s[len("sqlite:///"):]
            # if rest already begins with a slash, the user provided
            # sqlite:////absolute/path (rest starts with '/') — keep as-is
            if rest.startswith("/"):
                return s
            # otherwise, make it absolute and ensure four slashes
            abs_rest = os.path.abspath(rest)
            return f"sqlite:///{abs_rest}"
        return s
    # Expand environment variables and return a sqlite URI for plain paths.
    s = os.path.expandvars(s)
    if s.startswith("file:"):
        return s
    # Treat plain filesystem paths as sqlite DB backends for mlflow. If the
    # path is a directory, create it and place a `mlflow.db` file inside it.
    abs_path = os.path.abspath(s)
    try:
        if os.path.isdir(abs_path) or s.endswith("/"):
            os.makedirs(abs_path, exist_ok=True)
            db_path = os.path.join(abs_path, "mlflow.db")
        else:
            # If a file-like path was provided (ends with .db or has an ext),
            # use it directly as the sqlite DB file.
            parent = os.path.dirname(abs_path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            db_path = abs_path
        return f"sqlite:///{db_path}"
    except Exception:
        # Fallback: return sqlite URI for the absolute path
        return f"sqlite:///{abs_path}"


def _sanitize_mlflow_key(key: str) -> str:
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-. :/")
    return "".join(ch if ch in allowed else "_" for ch in str(key))


def _dataset_manifest(cfg_dict: Dict[str, Any]) -> Dict[str, Any]:
    data_cfg = cfg_dict.get("data", {}) if isinstance(cfg_dict.get("data"), dict) else {}
    return {
        "type": data_cfg.get("type"),
        "modalities": data_cfg.get("modalities"),
        "meshes": data_cfg.get("meshes"),
        "paths": data_cfg.get("paths"),
        "normalize": data_cfg.get("normalize"),
    }


def _namespace_to_dict(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _namespace_to_dict(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_namespace_to_dict(v) for v in obj]
    if isinstance(obj, SimpleNamespace):
        return {k: _namespace_to_dict(v) for k, v in obj.__dict__.items()}
    if hasattr(obj, "__dict__"):
        return {k: _namespace_to_dict(v) for k, v in obj.__dict__.items()}
    return obj


def _flatten_config(obj: Any, prefix: str = "") -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if isinstance(obj, dict):
        for key, value in obj.items():
            next_prefix = f"{prefix}.{key}" if prefix else str(key)
            out.update(_flatten_config(value, next_prefix))
        return out
    if isinstance(obj, list):
        out[prefix] = json.dumps(obj, sort_keys=True)
        return out
    out[prefix] = obj
    return out


def _output_context(verbose: bool):
    if verbose:
        return nullcontext()
    stack = ExitStack()
    stack.enter_context(redirect_stdout(io.StringIO()))
    stack.enter_context(redirect_stderr(io.StringIO()))
    return stack


def _get_train_epochs(cfg_dict: Dict[str, Any]) -> int:
    train_cfg = cfg_dict.get("train", {}) if isinstance(cfg_dict.get("train"), dict) else {}
    if "max_epochs" in train_cfg:
        return int(train_cfg.get("max_epochs") or 1)
    if "epochs" in train_cfg:
        return int(train_cfg.get("epochs") or 1)
    return 1


def _get_override_epochs(overrides: Optional[list[Dict[str, Any]]]) -> Optional[int]:
    if not overrides:
        return None
    for spec in overrides:
        path = str(spec.get("path", ""))
        if path in {"train.max_epochs", "train.epochs"}:
            try:
                return int(spec.get("value"))
            except Exception:
                return None
    return None


def run(cfg) -> None:
    # Expect the launcher config to provide `hpo.paths.base_config` (env-resolved).
    # Use that path directly; do not attempt multiple fallbacks.
    try:
        base_config_path = cfg.hpo.paths.base_config
    except Exception:
        base_config_path = getattr(getattr(cfg, "hpo", None), "base_config", None)

    if not base_config_path:
        raise ValueError("hpo.paths.base_config not found in launcher config; please set hpo.paths.base_config")

    cfg_dict = _load_config_raw(base_config_path)
    run_cfg = cfg_dict.get("run", {}) if isinstance(cfg_dict.get("run"), dict) else {}
    # Prefer the environment selector given to the HPO launcher (outer cfg.run.env).
    # Fall back to the base config's run.env when not provided.
    base_run_cfg = cfg_dict.get("run", {})
    outer_run = getattr(cfg, "run", None)
    if outer_run is not None:
        env_name = str(getattr(outer_run, "env", getattr(outer_run, "environment", None)))
    else:
        env_name = None
    if not env_name:
        env_name = str(base_run_cfg.get("env", base_run_cfg.get("environment", "local")))
    env_name = env_name.lower()
    cfg_dict = _resolve_env_select(cfg_dict, env_name)

    hpo_cfg = cfg.hpo
    study_name = str(getattr(hpo_cfg, "study_name", "cogedi_hpo"))
    direction = str(getattr(getattr(hpo_cfg, "objective", None), "direction", "minimize")).lower()
    optuna_cfg = getattr(hpo_cfg, "optuna", None)
    trials = int(getattr(optuna_cfg, "trials", 1)) if optuna_cfg is not None else 1
    n_jobs = int(getattr(hpo_cfg, "n_jobs", 1))
    # Use the launcher-provided output directory: `cfg.hpo.paths.output_dir`.
    try:
        out_dir = str(cfg.hpo.paths.output_dir)
    except Exception:
        out_dir = str(getattr(hpo_cfg, "output_dir", "hpo_runs"))
    exp_prefix = str(getattr(hpo_cfg, "experiment_prefix", "hpo"))
    resume = bool(getattr(hpo_cfg, "resume", False))
    verbose = bool(getattr(hpo_cfg, "verbose", True))
    eval_checkpoint = getattr(hpo_cfg, "eval_checkpoint", "checkpoint-latest.pth")
    mlflow_cfg = getattr(hpo_cfg, "mlflow", None)
    mlflow_enabled = bool(getattr(mlflow_cfg, "enabled", False)) if mlflow_cfg is not None else False
    # Use launcher-provided MLflow URI at `cfg.hpo.paths.mlflow_uri`.
    try:
        mlflow_uri = _resolve_mlflow_uri(cfg.hpo.paths.mlflow_uri)
    except Exception:
        mlflow_uri = _resolve_mlflow_uri(getattr(mlflow_cfg, "tracking_uri", None) if mlflow_cfg is not None else None)
    mlflow_experiment = study_name
    mlflow_run_prefix = str(getattr(mlflow_cfg, "run_prefix", exp_prefix)) if mlflow_cfg is not None else exp_prefix
    weighted_cfg = getattr(getattr(hpo_cfg, "metrics", None), "weighted", None)
    weighted_name = str(getattr(weighted_cfg, "name", "weighted_correspondence")) if weighted_cfg is not None else "weighted_correspondence"
    weighted_joint_key = str(getattr(weighted_cfg, "joint_key", "joint_correspondence_L2/B")) if weighted_cfg is not None else "joint_correspondence_L2/B"
    weighted_corr_key = str(getattr(weighted_cfg, "corr_key", "correspondence_evaluation_(average).mean_dist")) if weighted_cfg is not None else "correspondence_evaluation_(average).mean_dist"
    weighted_w = float(getattr(weighted_cfg, "weight", 0.5)) if weighted_cfg is not None else 0.5

    include = list(getattr(getattr(hpo_cfg, "metrics", None), "include", []))
    exclude = list(getattr(getattr(hpo_cfg, "metrics", None), "exclude", []))

    mode = str(getattr(hpo_cfg, "mode", "optuna")).lower()
    params = getattr(hpo_cfg, "params", None)
    overrides = getattr(hpo_cfg, "overrides", None)
    if params is None:
        raise ValueError("hpo.params is required")
    if not isinstance(params, list):
        raise ValueError("hpo.params must be a list of parameter specs")
    if overrides is not None and not isinstance(overrides, list):
        raise ValueError("hpo.overrides must be a list of override specs")
    params = [_spec_to_dict(spec, label="hpo.params") for spec in params]
    if overrides is not None:
        overrides = [_spec_to_dict(spec, label="hpo.overrides") for spec in overrides]
    if mode not in {"optuna", "ablation"}:
        raise ValueError("hpo.mode must be one of: optuna, ablation")

    ckpt_base = _get_attr(cfg_dict, "paths.checkpoints", None)
    if ckpt_base:
        study_ckpt_dir = os.path.join(ckpt_base, study_name)
        _set_attr(cfg_dict, "paths.checkpoints", study_ckpt_dir)

    rows: list[Dict[str, Any]] = []
    started_at = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(out_dir, exist_ok=True)

    if resume:
        latest_summary = _find_latest_summary(out_dir)
        rows = _load_csv(latest_summary)
        if rows:
            started_at = os.path.splitext(os.path.basename(latest_summary))[0].replace("hpo_summary_", "")

    register_all()

    mlflow = None
    if mlflow_enabled:
        try:
            import mlflow as _mlflow
        except ImportError as exc:
            raise ImportError("mlflow is required when hpo.mlflow.enabled is true") from exc
        mlflow = _mlflow
        # Normalize sqlite URIs to ensure absolute-file form (sqlite:////abs/path)
        if mlflow_uri and isinstance(mlflow_uri, str) and mlflow_uri.startswith("sqlite:"):
            # strip the 'sqlite:' prefix and ensure an absolute path
            rest = mlflow_uri[len("sqlite:"):]
            # If the remainder already starts with a slash, treat it as an
            # absolute path (possibly with multiple slashes) and preserve it.
            if rest.startswith("/"):
                path_part = "/" + rest.lstrip("/")
                abs_path = os.path.abspath(path_part)
            else:
                # relative path: make absolute relative to CWD
                abs_path = os.path.abspath(rest)
            # construct uri so that it becomes sqlite:////absolute/path
            mlflow_uri = f"sqlite:///{abs_path}"

        if mlflow_uri:
            mlflow.set_tracking_uri(mlflow_uri)
        # Determine a sensible default artifact root near the sqlite DB when
        # using file-backed sqlite. This prevents MLflow from creating local
        # `mlruns` or `hpo_runs` folders in the repo. Use the normalized
        # mlflow_uri (not mlflow.get_tracking_uri()) to avoid MLflow internal
        # rewriting affecting our decision.
        artifact_root = None
        resolved_tracking = mlflow_uri

        if resolved_tracking and isinstance(resolved_tracking, str) and resolved_tracking.startswith("sqlite:///"):
            db_path = resolved_tracking[len("sqlite:///") :]
            db_path = os.path.abspath(db_path)
            artifacts_dir = os.path.join(os.path.dirname(db_path), "artifacts")
            os.makedirs(artifacts_dir, exist_ok=True)
            # Use a standard file:// URI with an absolute path. Avoid adding
            # an extra leading slash which produces file:////home/... URIs.
            artifact_root = f"file://{os.path.abspath(artifacts_dir)}"

        # Create or set the experiment with the chosen artifact location
        try:
            existing = mlflow.get_experiment_by_name(mlflow_experiment)
            final_experiment_name = mlflow_experiment
            if existing is None:
                # Create experiment with our desired artifact root when possible
                if artifact_root:
                    mlflow.create_experiment(final_experiment_name, artifact_location=artifact_root)
                else:
                    mlflow.create_experiment(final_experiment_name)
            else:
                # If an existing experiment was found but its artifact_location
                # does not match our desired artifact_root, avoid reusing it
                # (which may point into the repo). Instead create a new
                # experiment with a timestamp suffix.
                try:
                    existing_art = existing.artifact_location or ""
                except Exception:
                    existing_art = ""
                if artifact_root and existing_art and os.path.abspath(existing_art).startswith("/"):
                    # existing_art may be a file:// URI or a plain path; normalize
                    norm_existing = existing_art
                    if norm_existing.startswith("file://"):
                        norm_existing = norm_existing[len("file://"):]
                    norm_existing = os.path.abspath(norm_existing)
                    norm_desired = artifact_root
                    if norm_desired.startswith("file://"):
                        norm_desired = norm_desired[len("file://"):]
                    norm_desired = os.path.abspath(norm_desired)
                    if not norm_existing.startswith(norm_desired):
                        final_experiment_name = f"{mlflow_experiment}_{started_at}"
                        mlflow.create_experiment(final_experiment_name, artifact_location=artifact_root)

            mlflow.set_experiment(final_experiment_name)
            mlflow_experiment = final_experiment_name
        except Exception:
            # Fallback to simple set_experiment if creation fails
            mlflow.set_experiment(mlflow_experiment)

        # Print diagnostic info so users can verify where MLflow will write.
        try:
            exp = mlflow.get_experiment_by_name(mlflow_experiment)
            exp_art = exp.artifact_location if exp is not None else None
        except Exception:
            exp_art = None
        print(f"[HPO] MLflow tracking URI: {mlflow.get_tracking_uri()}")
        print(f"[HPO] MLflow experiment: {mlflow_experiment} -> artifact_location={exp_art}")

    worker_count = max(1, n_jobs if mode == "optuna" else 1)
    bar_positions: queue.Queue[int] = queue.Queue()
    for pos in range(1, worker_count + 1):
        bar_positions.put(pos)
    bar_lock = threading.Lock()

    def _apply_overrides(trial_cfg: Dict[str, Any]) -> None:
        if not overrides:
            return
        for spec in overrides:
            path = str(spec.get("path"))
            value = spec.get("value")
            _set_attr(trial_cfg, path, value)

    def _log_mlflow_run(
        *,
        trial_id: str,
        trial_number: int,
        trial_params: Dict[str, Any],
        metrics: Dict[str, float],
        objective_val: float,
        report_path: str,
    ) -> None:
        if not mlflow:
            return

        run_name = f"{mlflow_run_prefix}_{trial_id}"
        with mlflow.start_run(run_name=run_name):
            mlflow.set_tag("trial_id", trial_id)
            mlflow.set_tag("trial_number", str(trial_number))
            mlflow.log_param("objective_metric", str(getattr(getattr(hpo_cfg, "objective", None), "metric", "")))
            mlflow.log_param("objective_direction", str(getattr(getattr(hpo_cfg, "objective", None), "direction", "")))

            for name, value in trial_params.items():
                mlflow.log_param(_sanitize_mlflow_key(f"param.{name}"), value)

            mlflow.log_metric("objective", objective_val)
            for key, value in metrics.items():
                mlflow.log_metric(_sanitize_mlflow_key(key), value)

            if report_path and os.path.exists(report_path):
                try:
                    mlflow.log_artifact(report_path, artifact_path="eval")
                except Exception as exc:
                    print(f"[HPO][MLFLOW] Warning: failed to log eval report {report_path}: {exc}")

            ckpt_base = _get_attr(cfg_dict, "paths.checkpoints", None)
            exp_name = _get_attr(cfg_dict, "run.experiment_name", None)
            if ckpt_base and exp_name:
                ckpt_dir = os.path.join(ckpt_base, exp_name)
                latest_ckpt = os.path.join(ckpt_dir, "checkpoint-latest.pth")
                if os.path.exists(latest_ckpt):
                    try:
                        mlflow.log_artifact(latest_ckpt, artifact_path="checkpoints")
                    except Exception as exc:
                        print(f"[HPO][MLFLOW] Warning: failed to log checkpoint {latest_ckpt}: {exc}")

            dataset_path = os.path.join(out_dir, f"dataset_manifest_{trial_id}.json")
            with open(dataset_path, "w", encoding="utf-8") as f:
                json.dump(_dataset_manifest(cfg_dict), f, indent=2, sort_keys=True)
            try:
                mlflow.log_artifact(dataset_path, artifact_path="data")
            except Exception as exc:
                print(f"[HPO][MLFLOW] Warning: failed to log dataset manifest {dataset_path}: {exc}")

            base_cfg_path = os.path.join(out_dir, f"base_config_{trial_id}.yaml")
            with open(base_cfg_path, "w", encoding="utf-8") as f:
                yaml.safe_dump(cfg_dict, f, sort_keys=False)
            try:
                mlflow.log_artifact(base_cfg_path, artifact_path="config")
            except Exception as exc:
                print(f"[HPO][MLFLOW] Warning: failed to log base config {base_cfg_path}: {exc}")

            hpo_cfg_path = os.path.join(out_dir, f"hpo_config_{trial_id}.yaml")
            with open(hpo_cfg_path, "w", encoding="utf-8") as f:
                yaml.safe_dump(_namespace_to_dict(hpo_cfg), f, sort_keys=False)
            try:
                mlflow.log_artifact(hpo_cfg_path, artifact_path="config")
            except Exception as exc:
                print(f"[HPO][MLFLOW] Warning: failed to log hpo config {hpo_cfg_path}: {exc}")

    def _run_trial(
        trial_cfg: Dict[str, Any],
        *,
        trial_id: str,
        trial_number: int,
        trial_params: Dict[str, Any],
    ) -> Tuple[float, Dict[str, float]]:
        with bar_lock:
            position = bar_positions.get()
        last_epoch = 0

        def _progress_hook(epoch_idx: int, _total: int) -> None:
            nonlocal last_epoch
            delta = int(epoch_idx) - last_epoch
            if delta > 0:
                trial_bar.update(delta)
                last_epoch = int(epoch_idx)

        try:
            _set_attr(trial_cfg, "run.experiment_name", trial_id)
            # Mark this trial as HPO-run and disable TensorBoard logging
            _set_attr(trial_cfg, "run._hpo", True)
            _set_attr(trial_cfg, "train.log_dir", None)

            # Resolve environment selectors in the trial config so path selectors
            # (e.g., paths.eval) become concrete strings for the current env.
            trial_cfg = _resolve_env_select(trial_cfg, env_name)

            seed_base = getattr(hpo_cfg, "trial_seed_base", None)
            if seed_base is not None:
                _set_attr(trial_cfg, "run.seed", int(seed_base) + int(trial_number))

            _apply_overrides(trial_cfg)
            train_epochs = _get_train_epochs(trial_cfg)
            override_epochs = _get_override_epochs(overrides)
            if override_epochs is not None:
                train_epochs = override_epochs
            trial_bar = tqdm(total=train_epochs + 1, desc=trial_id, position=position, leave=False)
            _set_attr(trial_cfg, "train._progress_hook", _progress_hook)

        # Train
            _set_attr(trial_cfg, "run.mode", "train")
            with _output_context(verbose):
                train.run(_to_namespace(trial_cfg))
            if last_epoch < train_epochs:
                trial_bar.update(train_epochs - last_epoch)
                last_epoch = train_epochs

        # Eval
            _set_attr(trial_cfg, "run.mode", "eval")
            if eval_checkpoint is not None:
                if str(eval_checkpoint).lower() == "latest":
                    _set_attr(trial_cfg, "eval.checkpoint", "checkpoint-latest.pth")
                else:
                    _set_attr(trial_cfg, "eval.checkpoint", eval_checkpoint)
            elif _get_attr(trial_cfg, "eval.checkpoint", None) is None:
                _set_attr(trial_cfg, "eval.checkpoint", "latest")
            with _output_context(verbose):
                eval.run(_to_namespace(trial_cfg))
            trial_bar.update(1)
        finally:
            try:
                trial_bar.close()
            except Exception:
                pass
            with bar_lock:
                bar_positions.put(position)

        eval_dir = _resolve_eval_output_dir(trial_cfg)
        report_path = os.path.join(eval_dir, "eval_report.txt") if eval_dir else ""
        metrics = _parse_eval_report(report_path)
        metrics = _filter_metrics(metrics, include=include, exclude=exclude)

        # If no metrics were produced by the eval run, don't blow up the whole
        # HPO process. Instead warn and return a worst-case objective so the
        # trial is recorded and optimization can continue.
        if not metrics:
            import warnings
            warnings.warn(
                f"HPO: no metrics parsed from eval report (report_path={report_path}, eval_dir={eval_dir})."
            )
            # Choose a conservative worst-case objective depending on direction
            if direction == "minimize":
                objective_val = float("inf")
            else:
                objective_val = float("-inf")

            row: Dict[str, Any] = {
                "trial": trial_number,
                "experiment": trial_id,
                "objective": objective_val,
            }
            row.update({f"param.{k}": v for k, v in trial_params.items()})
            row.update({f"metric.{k}": v for k, v in metrics.items()})
            rows.append(row)
            _write_csv(os.path.join(out_dir, f"hpo_summary_{started_at}.csv"), rows)

            _log_mlflow_run(
                trial_id=trial_id,
                trial_number=trial_number,
                trial_params=trial_params,
                metrics=metrics,
                objective_val=objective_val,
                report_path=report_path,
            )

            return objective_val, metrics

        if weighted_cfg is not None:
            joint_val = metrics.get(weighted_joint_key)
            corr_val = metrics.get(weighted_corr_key)
            if joint_val is None and not str(weighted_joint_key).startswith("joint_correspondence."):
                joint_val = metrics.get(f"joint_correspondence.{weighted_joint_key}")
            if corr_val is None and not str(weighted_corr_key).startswith("correspondence_evaluation_(average)."):
                corr_val = metrics.get(f"correspondence_evaluation_(average).{weighted_corr_key}")
            if joint_val is not None and corr_val is not None:
                metrics[weighted_name] = weighted_w * float(joint_val) + (1.0 - weighted_w) * float(corr_val)

        objective_key = str(getattr(getattr(hpo_cfg, "objective", None), "metric", "mean_dist"))

        # Try to be tolerant when the exact objective key isn't present in the
        # parsed metrics. Attempt substring/prefix matches, and if a weighted
        # objective was requested, try to construct it from available joint/corr
        # entries before giving up. If none of those succeed, raise a helpful
        # error that lists available metric keys for debugging.
        if objective_key not in metrics:
            # candidate keys that contain or end/start with the objective key
            candidates = [k for k in metrics.keys() if objective_key in k or k.endswith(objective_key) or k.startswith(objective_key)]
            if len(candidates) == 1:
                matched = candidates[0]
                objective_val = float(metrics[matched])
            else:
                constructed = False
                if weighted_cfg is not None and objective_key == weighted_name:
                    # try to build weighted metric from available joint/corr variants
                    joint_val = metrics.get(weighted_joint_key)
                    corr_val = metrics.get(weighted_corr_key)
                    # try alternate common prefixes
                    if joint_val is None:
                        alt = f"joint_correspondence.{weighted_joint_key}"
                        joint_val = metrics.get(alt, None)
                    if corr_val is None:
                        alt2 = f"correspondence_evaluation_(average).{weighted_corr_key}"
                        corr_val = metrics.get(alt2, None)
                    if joint_val is None or corr_val is None:
                        # try searching for any keys that look like joint/corr fragments
                        for k in metrics.keys():
                            if "joint_correspond" in k and joint_val is None:
                                joint_val = metrics[k]
                            if "correspondence_evaluation" in k and corr_val is None:
                                corr_val = metrics[k]
                    if joint_val is not None and corr_val is not None:
                        metrics[weighted_name] = weighted_w * float(joint_val) + (1.0 - weighted_w) * float(corr_val)
                        objective_val = float(metrics[weighted_name])
                        constructed = True

                if not constructed:
                    avail = sorted(metrics.keys())
                    raise ValueError(
                        f"Objective metric '{objective_key}' not found in eval report. "
                        f"Available metrics: {avail}"
                    )
        else:
            objective_val = float(metrics[objective_key])

        row: Dict[str, Any] = {
            "trial": trial_number,
            "experiment": trial_id,
            "objective": objective_val,
        }
        row.update({f"param.{k}": v for k, v in trial_params.items()})
        row.update({f"metric.{k}": v for k, v in metrics.items()})
        rows.append(row)
        _write_csv(os.path.join(out_dir, f"hpo_summary_{started_at}.csv"), rows)

        _log_mlflow_run(
            trial_id=trial_id,
            trial_number=trial_number,
            trial_params=trial_params,
            metrics=metrics,
            objective_val=objective_val,
            report_path=report_path,
        )

        return objective_val, metrics

    if mode == "optuna":
        # Prefer launcher-provided optuna storage at `cfg.hpo.paths.optuna_storage`
        storage = None
        try:
            storage = _resolve_mlflow_uri(cfg.hpo.paths.optuna_storage)
        except Exception:
            storage = None
        if storage is None:
            storage = _get_attr(cfg_dict, "hpo.optuna.storage", None)
        grid_max_out = bool(_get_attr(cfg_dict, "hpo.optuna.grid_max_out", False))
        if resume and not storage:
            # Previously this raised an error. Allow resuming to be a best-effort
            # when no persistent storage is configured: start a new study instead.
            print("hpo.resume requested but no hpo.study.storage configured; starting a new study")
            resume = False

        # Build sampler/pruner once so they can be reused whether loading or
        # creating the study.
        sampler = _build_sampler(cfg_dict, params)
        pruner = _build_pruner(cfg_dict)

        if resume and storage:
            try:
                study = optuna.load_study(study_name=study_name, storage=storage)
            except KeyError:
                # Study not found in storage: create it so resume becomes
                # idempotent (user asked to resume but storage has no record).
                study = optuna.create_study(
                    study_name=study_name,
                    direction=direction,
                    sampler=sampler,
                    pruner=pruner,
                    storage=storage,
                    load_if_exists=True,
                )
        else:
            study = optuna.create_study(
                study_name=study_name,
                direction=direction,
                sampler=sampler,
                pruner=pruner,
                storage=storage,
                load_if_exists=bool(storage),
            )

        sampler_name = str(_get_attr(cfg_dict, "hpo.optuna.sampler", "tpe")).lower()
        if sampler_name == "grid" and grid_max_out:
            grid = _get_attr(cfg_dict, "hpo.optuna.grid_search_space", None)
            if grid is None:
                grid = _build_grid_search_space(params)
            total = 1
            for values in grid.values():
                total *= len(values)
            trials = total

        def _objective(trial: optuna.Trial) -> float:
            trial_cfg = deepcopy(cfg_dict)
            trial_id = f"{exp_prefix}_t{trial.number:04d}"

            trial_params = {}
            for spec in params:
                name = str(spec.get("name"))
                path = str(spec.get("path"))
                value = _suggest_value(trial, spec)
                _set_attr(trial_cfg, path, value)
                trial_params[name] = value

            objective_val, _ = _run_trial(
                trial_cfg,
                trial_id=trial_id,
                trial_number=trial.number,
                trial_params=trial_params,
            )
            return objective_val

        study_bar = tqdm(total=trials, desc="HPO", position=0)

        def _callback(_study: optuna.Study, _trial: optuna.Trial) -> None:
            study_bar.update(1)

        try:
            study.optimize(_objective, n_trials=trials, n_jobs=n_jobs, callbacks=[_callback])
        finally:
            study_bar.close()

        print("HPO completed")
        print(f"Best value: {study.best_value}")
        print(f"Best params: {study.best_params}")
        return

    include_baseline = bool(getattr(getattr(hpo_cfg, "ablation", None), "include_baseline", True))
    total_ablation = 0
    if include_baseline:
        total_ablation += 1
    for spec in params:
        total_ablation += len(_ablation_values(spec))
    study_bar = tqdm(total=total_ablation, desc="HPO", position=0)
    trial_counter = 0
    best_value = None
    best_trial = None

    existing_trials = {str(row.get("experiment", "")) for row in rows}
    for row in rows:
        try:
            obj_val = float(row.get("objective"))
        except Exception:
            continue
        if best_value is None:
            best_value = obj_val
            best_trial = row.get("experiment")
            continue
        if direction == "minimize" and obj_val < best_value:
            best_value = obj_val
            best_trial = row.get("experiment")
        if direction == "maximize" and obj_val > best_value:
            best_value = obj_val
            best_trial = row.get("experiment")

    if include_baseline:
        trial_cfg = deepcopy(cfg_dict)
        trial_id = f"{exp_prefix}_baseline"
        if trial_id not in existing_trials:
            objective_val, _ = _run_trial(
                trial_cfg,
                trial_id=trial_id,
                trial_number=trial_counter,
                trial_params={"ablation": "baseline"},
            )
            best_value = objective_val if best_value is None else best_value
            best_trial = trial_id if best_trial is None else best_trial
            trial_counter += 1
            study_bar.update(1)

    for spec in params:
        name = str(spec.get("name"))
        path = str(spec.get("path"))
        for value in _ablation_values(spec):
            trial_cfg = deepcopy(cfg_dict)
            _set_attr(trial_cfg, path, value)
            trial_id = f"{exp_prefix}_{name}_{trial_counter:04d}"
            if trial_id in existing_trials:
                trial_counter += 1
                study_bar.update(1)
                continue
            objective_val, _ = _run_trial(
                trial_cfg,
                trial_id=trial_id,
                trial_number=trial_counter,
                trial_params={"ablation": name, name: value},
            )
            if best_value is None:
                best_value = objective_val
                best_trial = trial_id
            else:
                if direction == "minimize" and objective_val < best_value:
                    best_value = objective_val
                    best_trial = trial_id
                if direction == "maximize" and objective_val > best_value:
                    best_value = objective_val
                    best_trial = trial_id
            trial_counter += 1
            study_bar.update(1)

    study_bar.close()

    print("HPO ablation completed")
    print(f"Best value: {best_value}")
    print(f"Best trial: {best_trial}")
