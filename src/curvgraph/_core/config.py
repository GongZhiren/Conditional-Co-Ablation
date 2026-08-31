from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


def _read_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data if data is not None else {}


def load_config(default_config_path: str) -> Dict[str, Any]:
    root_path = Path(default_config_path).resolve()
    root_cfg = _read_yaml(root_path)
    includes = root_cfg.get("includes", {})
    merged: Dict[str, Any] = dict(root_cfg)

    for key, include_path in includes.items():
        include_file = Path(include_path)
        if not include_file.is_absolute():
            candidate_local = (root_path.parent / include_file).resolve()
            candidate_project = (root_path.parent.parent / include_file).resolve()
            if candidate_local.exists():
                include_file = candidate_local
            else:
                include_file = candidate_project
        include_data = _read_yaml(include_file)
        merged[key] = include_data
    merged["_config_root"] = str(root_path.parent)
    model_section = merged.get("model", {})
    models = model_section.get("models", {})
    for _model_key, spec in models.items():
        path = spec.get("path")
        if isinstance(path, str):
            candidate = Path(path)
            if not candidate.is_absolute():
                candidate_local = (root_path.parent / candidate).resolve()
                candidate_project = (root_path.parent.parent / candidate).resolve()
                if candidate_local.exists():
                    spec["path"] = str(candidate_local)
                elif candidate_project.exists():
                    spec["path"] = str(candidate_project)
    return merged


def validate_config(cfg: Dict[str, Any]) -> None:
    required_top = ["project", "run", "model"]
    missing = [k for k in required_top if k not in cfg]
    if missing:
        raise ValueError(f"Missing top-level config sections: {missing}")

    model_key = cfg["run"]["model_key"]
    models = cfg["model"].get("models", {})
    if model_key not in models:
        raise ValueError(f"Unknown model_key={model_key}, available={list(models)}")

    model_cfg = models[model_key]
    family = str(model_cfg.get("family", ""))
    allow_moe = bool(model_cfg.get("allow_moe", False))
    if family == "mixtral" and not allow_moe:
        raise ValueError("Mixtral/MoE is disabled by default. Set allow_moe=true explicitly to run.")


def model_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    model_key = cfg["run"]["model_key"]
    return cfg["model"]["models"][model_key]
