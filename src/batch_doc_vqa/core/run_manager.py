#!/usr/bin/env python3
"""
Run management system for batch document VQA experiments.
Creates dated run directories with proper configuration tracking.
"""
import yaml
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
import re
import subprocess
import hashlib


REPRO_RELEVANT_PATH_PREFIXES = (
    "src/",
    "imgs/",
    "tests/data/",
)

REPRO_RELEVANT_PATH_EXACT = {
    "pyproject.toml",
    "uv.lock",
    "requirements.txt",
}


def _to_json_safe(value: Any) -> Any:
    """Convert values to JSON-safe structures with stable ordering."""
    if isinstance(value, dict):
        return {str(k): _to_json_safe(v) for k, v in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (list, tuple)):
        return [_to_json_safe(v) for v in value]
    if isinstance(value, set):
        return sorted(_to_json_safe(v) for v in value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _stable_hash(payload: Any) -> str:
    """Create a stable SHA256 hash for arbitrary JSON-serializable payloads."""
    normalized = _to_json_safe(payload)
    raw = json.dumps(normalized, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _safe_run_git_command(args: List[str]) -> Optional[str]:
    """Run a git command safely and return stripped stdout when available."""
    try:
        result = subprocess.run(
            ["git", *args],
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except Exception:
        return None

    output = result.stdout.strip()
    return output if output else None


def _dequote_git_path(path: str) -> str:
    """Best-effort dequote for porcelain paths with quoted whitespace/escapes."""
    if len(path) >= 2 and path[0] == '"' and path[-1] == '"':
        raw = path[1:-1]
        try:
            return bytes(raw, "utf-8").decode("unicode_escape")
        except Exception:
            return raw
    return path


def _extract_git_dirty_paths(porcelain_output: Optional[str]) -> Optional[List[str]]:
    """Extract normalized file paths from `git status --porcelain` output."""
    if porcelain_output is None:
        return None

    paths: List[str] = []
    for raw_line in porcelain_output.splitlines():
        line = raw_line.rstrip()
        if len(line) < 4:
            continue

        payload = line[3:].strip()
        if not payload:
            continue

        # Rename/copy format: "old/path -> new/path" (keep destination path).
        if " -> " in payload:
            payload = payload.split(" -> ", 1)[1].strip()

        normalized = _dequote_git_path(payload).replace("\\", "/").strip()
        if normalized:
            paths.append(normalized)

    return paths


def _is_reproducibility_relevant_dirty_path(path: str) -> bool:
    """Return whether a dirty path likely impacts inference/benchmark reproducibility."""
    if path in REPRO_RELEVANT_PATH_EXACT:
        return True
    return any(path.startswith(prefix) for prefix in REPRO_RELEVANT_PATH_PREFIXES)


def build_git_dirty_warning_lines(config: "RunConfig", *, max_paths: int = 6) -> List[str]:
    """Build pre-run warning/info lines for dirty worktree status."""
    if config.git_dirty_relevant:
        total_relevant = config.git_dirty_relevant_count or len(config.git_dirty_relevant_paths)
        lines = [
            "[yellow]⚠️ Reproducibility warning: relevant uncommitted changes detected.[/yellow]",
            f"[yellow]Relevant dirty paths: {total_relevant}[/yellow]",
        ]
        for path in config.git_dirty_relevant_paths[:max_paths]:
            lines.append(f"[yellow]- {path}[/yellow]")
        remaining = total_relevant - min(max_paths, len(config.git_dirty_relevant_paths))
        if remaining > 0:
            lines.append(f"[yellow]... and {remaining} more[/yellow]")
        lines.append("[yellow]Commit/stash relevant changes for strict run-to-run comparability.[/yellow]")
        return lines

    if config.git_dirty_raw:
        return [
            "[dim]ℹ️ Uncommitted changes detected, but none matched reproducibility-relevant paths.[/dim]"
        ]

    return []


class RunConfig:
    """Configuration for a model run."""
    def __init__(self, 
                 org: str,
                 model: str,
                 variant: Optional[str] = None,
                 model_size: Optional[str] = None,
                 open_weights: bool = False,
                 license_info: str = "Unknown",
                 api_provider: Optional[str] = None,
                 use_structured_output: bool = False,
                 use_regex_patterns: bool = False,
                 temperature: float = 0.0,
                 max_tokens: Optional[int] = None,
                 runtime_environment: str = "Unknown",
                 parser_version: str = "v1",
                 schema_version: str = "v1",
                 additional_config: Optional[Dict[str, Any]] = None):
        
        self.org = org
        self.model = model
        self.variant = variant
        self.model_size = model_size
        self.open_weights = open_weights
        self.license_info = license_info
        self.api_provider = api_provider
        self.use_structured_output = use_structured_output
        self.use_regex_patterns = use_regex_patterns
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.runtime_environment = runtime_environment
        self.parser_version = parser_version
        self.schema_version = schema_version
        self.additional_config = additional_config or {}
        
        # Generate run timestamp
        self.timestamp = datetime.utcnow()
        self.timestamp_str = self.timestamp.strftime("%Y%m%d_%H%M%S")
        
        # Generate run directory name
        self.run_name = self._generate_run_name()

        # Reproducibility metadata used for cohorting and diagnostics
        self.git_commit = _safe_run_git_command(["rev-parse", "HEAD"]) or "unknown"
        git_status = _safe_run_git_command(["status", "--porcelain"])
        dirty_paths = _extract_git_dirty_paths(git_status)
        self.git_dirty = bool(dirty_paths) if dirty_paths is not None else None
        self.git_dirty_raw = self.git_dirty
        self.git_dirty_raw_count = len(dirty_paths) if dirty_paths is not None else None
        if dirty_paths is None:
            self.git_dirty_relevant = None
            self.git_dirty_relevant_count = None
            self.git_dirty_relevant_paths: List[str] = []
        else:
            relevant_paths = [p for p in dirty_paths if _is_reproducibility_relevant_dirty_path(p)]
            self.git_dirty_relevant = bool(relevant_paths)
            self.git_dirty_relevant_count = len(relevant_paths)
            self.git_dirty_relevant_paths = relevant_paths
        self.prompt_hash = self._compute_prompt_hash()
        self.inference_settings_hash = self._compute_inference_settings_hash()
        
    def _generate_run_name(self) -> str:
        """Generate a standardized run directory name."""
        parts = [self.org, self.model]
        if self.variant:
            parts.append(self.variant)
        
        name = "-".join(parts).replace("/", "-").replace("_", "-")
        return f"{name}_{self.timestamp_str}"

    def _compute_prompt_hash(self) -> Optional[str]:
        """Hash the effective prompt template if available."""
        prompt_candidates = (
            self.additional_config.get("prompt_template"),
            self.additional_config.get("prompt"),
        )
        for candidate in prompt_candidates:
            if isinstance(candidate, str) and candidate.strip():
                return hashlib.sha256(candidate.encode("utf-8")).hexdigest()
        return None

    def _compute_inference_settings_hash(self) -> str:
        """Hash inference settings excluding the raw prompt text."""
        additional_without_prompt = {
            key: value
            for key, value in self.additional_config.items()
            if key not in {"prompt_template", "prompt"}
        }
        payload = {
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "structured_output": self.use_structured_output,
            "regex_patterns": self.use_regex_patterns,
            "additional": additional_without_prompt,
        }
        return _stable_hash(payload)

    def _extract_provider_routing_config(self) -> Dict[str, Any]:
        """Extract provider-routing-relevant config fields for reproducibility metadata."""
        routing_fields = {}
        for key, value in self.additional_config.items():
            lowered = str(key).lower()
            if "provider" in lowered or "route" in lowered or "fallback" in lowered:
                routing_fields[key] = value
        return _to_json_safe(routing_fields)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for YAML serialization."""
        # Derive routing metadata at serialization time so late-bound runtime
        # fields (e.g. discovered providers) stay consistent in config/manifest.
        provider_routing_config = self._extract_provider_routing_config()
        config_dict = {
            "run_info": {
                "run_name": self.run_name,
                "timestamp": self.timestamp_str,
                "timestamp_iso": self.timestamp.isoformat(),
                "reproducibility": {
                    "manifest_version": "1",
                    "git_commit": self.git_commit,
                    "git_dirty_raw": self.git_dirty_raw,
                    "git_dirty_raw_count": self.git_dirty_raw_count,
                    "git_dirty_relevant": self.git_dirty_relevant,
                    "git_dirty_relevant_count": self.git_dirty_relevant_count,
                    "git_dirty_relevant_paths_sample": self.git_dirty_relevant_paths[:10],
                    "git_dirty": self.git_dirty,
                    "prompt_hash": self.prompt_hash,
                    "parser_version": self.parser_version,
                    "schema_version": self.schema_version,
                    "inference_settings_hash": self.inference_settings_hash,
                    "provider_routing_config": provider_routing_config,
                },
            },
            "model": {
                "org": self.org,
                "model": self.model,
                "variant": self.variant,
                "model_size": self.model_size,
                "open_weights": self.open_weights,
                "license_info": self.license_info,
            },
            "api": {
                "provider": self.api_provider,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
            },
            "features": {
                "structured_output": self.use_structured_output,
                "regex_patterns": self.use_regex_patterns,
            },
            "environment": {
                "runtime": self.runtime_environment,
            },
            "additional": self.additional_config
        }
        return config_dict


class RunManager:
    """Manages experiment runs and their organization."""
    
    def __init__(self, base_output_dir: str = "tests/output/runs"):
        self.base_output_dir = Path(base_output_dir)
        self.base_output_dir.mkdir(parents=True, exist_ok=True)
    
    def create_run_directory(self, config: RunConfig) -> Path:
        """Create a new run directory with the given configuration."""
        run_dir = self.base_output_dir / config.run_name
        run_dir.mkdir(parents=True, exist_ok=True)

        # Save configuration and reproducibility manifest
        self.save_run_config(config.run_name, config.to_dict())

        return run_dir

    def _write_manifest(self, run_dir: Path, config: Dict[str, Any]) -> Path:
        """Write a lightweight reproducibility manifest for a run."""
        run_info = config.get("run_info", {})
        repro = run_info.get("reproducibility", {})
        manifest = {
            "manifest_version": repro.get("manifest_version", "1"),
            "run_name": run_info.get("run_name", run_dir.name),
            "timestamp": run_info.get("timestamp"),
            "timestamp_iso": run_info.get("timestamp_iso"),
            "model": config.get("model", {}),
            "api": config.get("api", {}),
            "reproducibility": repro,
            "artifacts": {
                "config": "config.yaml",
                "results": "results.json",
                "table_results": "table_results.json",
            },
        }
        manifest_path = run_dir / "manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(_to_json_safe(manifest), f, indent=2)
        return manifest_path

    def save_run_config(self, run_name: str, config: Dict[str, Any]) -> Path:
        """Save run configuration and refresh manifest metadata."""
        run_dir = self.base_output_dir / run_name
        if not run_dir.exists():
            raise ValueError(f"Run directory {run_name} does not exist")

        config_path = run_dir / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(_to_json_safe(config), f, default_flow_style=False, sort_keys=False)

        self._write_manifest(run_dir, config)
        return config_path
    
    def list_runs(self, pattern: Optional[str] = None) -> List[Dict[str, Any]]:
        """List all runs, optionally filtered by pattern."""
        runs = []
        
        for run_dir in self.base_output_dir.iterdir():
            if not run_dir.is_dir():
                continue
            
            if pattern and not re.search(pattern, run_dir.name, re.IGNORECASE):
                continue
                
            config_path = run_dir / "config.yaml"
            if config_path.exists():
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
                
                # Check for results
                results_path = run_dir / "results.json"
                has_results = results_path.exists()
                
                # Check for table results  
                table_results_path = run_dir / "table_results.json"
                has_table_results = table_results_path.exists()
                
                manifest_path = run_dir / "manifest.json"
                has_manifest = manifest_path.exists()
                
                run_info = {
                    "run_name": run_dir.name,
                    "run_dir": str(run_dir),
                    "config": config,
                    "has_results": has_results,
                    "has_table_results": has_table_results,
                    "has_manifest": has_manifest,
                }
                runs.append(run_info)
        
        # Sort by timestamp (newest first)
        runs.sort(key=lambda x: x["config"]["run_info"]["timestamp"], reverse=True)
        return runs
    
    def get_latest_run(self, pattern: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Get the latest run matching the pattern."""
        runs = self.list_runs(pattern)
        return runs[0] if runs else None
    
    def save_results(self, run_name: str, results: Dict[str, Any]) -> Path:
        """Save inference results to a run directory."""
        run_dir = self.base_output_dir / run_name
        if not run_dir.exists():
            raise ValueError(f"Run directory {run_name} does not exist")
        
        results_path = run_dir / "results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        return results_path
    
    def save_table_results(self, run_name: str, table_results: Dict[str, Any]) -> Path:
        """Save computed table results to a run directory."""
        run_dir = self.base_output_dir / run_name
        if not run_dir.exists():
            raise ValueError(f"Run directory {run_name} does not exist")
        
        table_results_path = run_dir / "table_results.json"
        with open(table_results_path, 'w') as f:
            json.dump(table_results, f, indent=2)
        
        return table_results_path
    
    def load_results(self, run_name: str) -> Dict[str, Any]:
        """Load results from a run directory."""
        results_path = self.base_output_dir / run_name / "results.json"
        if not results_path.exists():
            raise FileNotFoundError(f"Results file not found in run {run_name}")
        
        with open(results_path, 'r') as f:
            return json.load(f)
    
    def load_table_results(self, run_name: str) -> Dict[str, Any]:
        """Load table results from a run directory."""
        table_results_path = self.base_output_dir / run_name / "table_results.json"
        if not table_results_path.exists():
            raise FileNotFoundError(f"Table results file not found in run {run_name}")
        
        with open(table_results_path, 'r') as f:
            return json.load(f)


def create_glm_run_example():
    """Create an example run configuration for GLM-4.5V."""
    config = RunConfig(
        org="z-ai",
        model="glm-4.5v",
        model_size="Unknown",
        open_weights=False,
        license_info="Proprietary",
        api_provider="OpenRouter",
        use_structured_output=True,
        use_regex_patterns=False,
        temperature=0.0,
        max_tokens=2048,
        runtime_environment="OpenRouter API",
        additional_config={
            "base_url": "https://openrouter.ai/api/v1/chat/completions",
            "endpoint_type": "chat_completions",
            "response_format": "json"
        }
    )
    return config


if __name__ == "__main__":
    # Example usage
    manager = RunManager()
    
    # Create example run
    config = create_glm_run_example()
    run_dir = manager.create_run_directory(config)
    print(f"Created run directory: {run_dir}")
    
    # List runs
    runs = manager.list_runs()
    print(f"\nFound {len(runs)} runs:")
    for run in runs[:3]:  # Show first 3
        print(f"  {run['run_name']} - {run['config']['model']['org']}/{run['config']['model']['model']}")
