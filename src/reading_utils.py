import json
from pathlib import Path
from typing import Any


def _load_metadata(path: Path, purpose: str) -> Any:
    with path.open("rt", encoding="utf-8") as fp:
        raw = json.load(fp)

    if isinstance(raw, dict):
        entry = raw.get(purpose)
        if isinstance(entry, dict):
            return entry
        if entry is not None:
            return entry
        # Legacy format: metadata is not split by purpose.
        return raw

    msg = f"Unexpected metadata structure in {path}: expected a JSON object."
    raise TypeError(msg)


def _unique_paths(paths: list[Path]) -> list[Path]:
    seen: set[str] = set()
    ordered: list[Path] = []
    for candidate in paths:
        key = candidate.as_posix()
        if candidate.exists():
            key = candidate.resolve().as_posix()
        if key in seen:
            continue
        seen.add(key)
        ordered.append(candidate)
    return ordered


def read_metadata(data_path: str, purpose: str, file_name: str = "metadata.json"):
    data_root = Path(data_path).expanduser()
    requested = Path(file_name)

    candidates: list[Path] = []
    if requested.is_absolute():
        candidates.append(requested)
    else:
        candidates.append(data_root / requested)

    # Allow a couple of alternative layouts that appear in public datasets.
    candidates.extend(
        [
            data_root / "metadata.json",
            data_root / f"metadata_{purpose}.json",
            data_root / f"metadata-{purpose}.json",
            data_root / purpose / "metadata.json",
            data_root / "metadata" / "metadata.json",
            data_root / "metadata" / f"{purpose}.json",
        ]
    )

    if data_root.exists():
        # Add any other direct matches in the directory as a last resort.
        candidates.extend(sorted(data_root.glob("metadata*.json")))

    errors: list[str] = []
    for path in _unique_paths(candidates):
        if not path.exists() or path.is_dir():
            continue
        try:
            return _load_metadata(path, purpose)
        except (json.JSONDecodeError, TypeError) as exc:
            errors.append(f"{path}: {exc}")

    available = []
    if data_root.exists():
        available = sorted(p.as_posix() for p in data_root.glob("metadata*.json"))

    tried = ", ".join(p.as_posix() for p in _unique_paths(candidates))
    msg = (
        f"Metadata for '{purpose}' was not found. "
        f"Search root: {data_root.as_posix()} | Tried: {tried or '—'}"
    )
    if available:
        msg += f" | Available metadata files: {', '.join(available)}"
    if errors:
        msg += f" | Encountered errors: {', '.join(errors)}"
    raise FileNotFoundError(msg)


def flags_to_dict(FLAGS):
    flags_dict = {}
    for name in FLAGS:
        flag_value = FLAGS[name].value
        flags_dict[name] = flag_value
    return flags_dict
