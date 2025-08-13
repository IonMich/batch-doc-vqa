#!/usr/bin/env python3
"""
Run management system for batch document VQA experiments.
Creates dated run directories with proper configuration tracking.
"""
import os
import yaml
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
import re


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
        self.additional_config = additional_config or {}
        
        # Generate run timestamp
        self.timestamp = datetime.utcnow()
        self.timestamp_str = self.timestamp.strftime("%Y%m%d_%H%M%S")
        
        # Generate run directory name
        self.run_name = self._generate_run_name()
        
    def _generate_run_name(self) -> str:
        """Generate a standardized run directory name."""
        parts = [self.org, self.model]
        if self.variant:
            parts.append(self.variant)
        
        name = "-".join(parts).replace("/", "-").replace("_", "-")
        return f"{name}_{self.timestamp_str}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for YAML serialization."""
        config_dict = {
            "run_info": {
                "run_name": self.run_name,
                "timestamp": self.timestamp_str,
                "timestamp_iso": self.timestamp.isoformat(),
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
        
        # Save configuration
        config_path = run_dir / "config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config.to_dict(), f, default_flow_style=False, sort_keys=False)
        
        return run_dir
    
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
                
                run_info = {
                    "run_name": run_dir.name,
                    "run_dir": str(run_dir),
                    "config": config,
                    "has_results": has_results,
                    "has_table_results": has_table_results,
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