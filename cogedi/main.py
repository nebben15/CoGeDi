### main script logic --- reads config and runs appropriate subfunction ###
import argparse
import os
import yaml
from pathlib import Path
from types import SimpleNamespace

from cogedi.registry import register_all
from cogedi.build import build_eval
import cogedi.orch.train as train
import cogedi.orch.sample as sample
import cogedi.orch.inverse as inverse
import cogedi.orch.eval as eval
import cogedi.orch.info as info
import cogedi.orch.hpo as hpo

def get_args_parser():
    parser = argparse.ArgumentParser('CoGeDi', add_help=False)
    parser.add_argument('--config', required=True, type=str, help='Path to YAML config file')
    return parser

def to_namespace(obj):
    """Recursively convert nested dict/list structures to dot-access namespaces."""
    if isinstance(obj, dict):
        return SimpleNamespace(**{k: to_namespace(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [to_namespace(v) for v in obj]
    return obj

def load_config_raw(path):
    with open(path, 'r') as f:
        cfg = yaml.safe_load(f)
    if cfg is None:
        raise ValueError("Empty config file")
    return cfg

def _resolve_env_select(obj, env_name: str):
    """Resolve {local: ..., slurm: ...} style config blocks and expand env vars."""
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

def print_config(cfg_dict):
    print("\n" + "=" * 80)
    print("CONFIGURATION")
    print("=" * 80)
    print(yaml.dump(cfg_dict, sort_keys=False, default_flow_style=False))
    print("=" * 80 + "\n")

def main(args):
    # use one central config
    cfg_dict = load_config_raw(args.config)
    run_cfg = cfg_dict.get("run", {}) if isinstance(cfg_dict, dict) else {}
    env_name = str(run_cfg.get("env", run_cfg.get("environment", "local"))).lower()
    cfg_dict = _resolve_env_select(cfg_dict, env_name)
    print_config(cfg_dict)
    cfg = to_namespace(cfg_dict)

    # set up registry for building
    register_all()

    # choose from train, eval, sample, inverse, or info mode
    if cfg.run.mode == 'train':
        train.run(cfg)
    elif cfg.run.mode == 'sample':
        sample.run(cfg)
    elif cfg.run.mode == 'inverse':
        inverse.run(cfg)
    elif cfg.run.mode == 'eval':
        eval.run(cfg)
    elif cfg.run.mode == 'info':
        info.run(cfg)
    elif cfg.run.mode == 'hpo':
        hpo.run(cfg)
    else:
        raise ValueError("Mode not in [train|sample|inverse|eval|info|hpo]")

if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)