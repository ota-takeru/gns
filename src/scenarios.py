"""
Scenario registry utilities to support multiple training environments.

The registry enables selecting between pre-defined simulation setups
such as rigid bodies and fluids while keeping the design open for new
cases without touching the training loop.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping


@dataclass(frozen=True)
class Scenario:
    """Container describing a training scenario/dataset bundle."""

    key: str
    dataset_dir: Path
    metadata_split: str = "train"
    rollout_dataset: str | None = None
    rollout_metadata_split: str | None = None
    description: str | None = None
    overrides: Dict[str, Any] = field(default_factory=dict)

    def apply_overrides(self, cfg: Any) -> None:
        """Apply scenario-specific config overrides onto the loader config."""
        for attr, value in self.overrides.items():
            if not hasattr(cfg, attr):
                msg = f"Unsupported override '{attr}' for scenario '{self.key}'."
                raise AttributeError(msg)
            setattr(cfg, attr, value)


class ScenarioRegistry:
    """Registry that resolves scenario keys to Scenario instances."""

    def __init__(
        self,
        default_dataset: Path,
        custom_map: Mapping[str, Mapping[str, Any]] | None = None,
        default_keys: Iterable[str] | None = None,
    ) -> None:
        self._default_dataset = default_dataset
        self._scenarios: Dict[str, Scenario] = {}
        aliases = list(dict.fromkeys(default_keys or ("fluid", "rigid")))
        for alias in aliases:
            self.register(alias, {"dataset": default_dataset})
        if custom_map:
            for key, spec in custom_map.items():
                self.register(key, spec)

    def register(self, key: str, spec: Mapping[str, Any]) -> None:
        dataset_value = spec.get("dataset") or spec.get("data_path") or spec.get("path")
        dataset_dir = (
            Path(dataset_value).expanduser().resolve()
            if dataset_value
            else self._default_dataset
        )
        metadata_split = spec.get("metadata_split", "train")
        rollout_dataset = spec.get("rollout_dataset")
        rollout_metadata_split = spec.get("rollout_metadata_split")
        description = spec.get("description")
        overrides = dict(spec.get("overrides", {}))
        scenario = Scenario(
            key=key,
            dataset_dir=dataset_dir,
            metadata_split=metadata_split,
            rollout_dataset=rollout_dataset,
            rollout_metadata_split=rollout_metadata_split,
            description=description,
            overrides=overrides,
        )
        self._scenarios[key] = scenario

    def get(self, key: str) -> Scenario:
        if key in self._scenarios:
            return self._scenarios[key]

        # Allow users to pass a direct dataset path as scenario identifier.
        candidate_path = Path(key)
        if candidate_path.exists():
            scenario = Scenario(key=key, dataset_dir=candidate_path.resolve())
            self._scenarios[key] = scenario
            return scenario

        available = ", ".join(sorted(self._scenarios))
        msg = f"Scenario '{key}' is not defined. Available options: {available}"
        raise KeyError(msg)

    def available(self) -> Dict[str, Scenario]:
        return dict(self._scenarios)
