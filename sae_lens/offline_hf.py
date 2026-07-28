"""
Offline-aware wrapper around `datasets.load_dataset`.

When HF offline env vars are set (HF_HUB_OFFLINE / HF_DATASETS_OFFLINE /
TRANSFORMERS_OFFLINE), the normal `load_dataset(repo_id, ...)` path still tries
to contact the Hub to resolve dataset metadata and fails with
`OfflineModeIsEnabled` even if the parquet shards are present in the cache.

This module intercepts that case: it locates the cached parquet shards inside
the HF hub cache layout and loads them via the built-in `parquet` builder,
which needs no network. In online mode the wrapper is a transparent pass-through.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from datasets import load_dataset

from sae_lens import logger


def _register_zstd_with_fsspec() -> None:
    """fsspec ships a placeholder entry for 'zstd' but no decoder even when
    `zstandard` is installed. Datasets that stream .jsonl.zst shards (e.g.
    monology/pile-uncopyrighted) then fail with
    ``ValueError: Compression type zstd not supported``. Wire zstandard in
    once per process."""
    try:
        import fsspec.compression as _fc
        import zstandard as _zstd
    except Exception:
        return
    if callable(_fc.compr.get("zstd")):
        return

    def _zstd_reader(infile, mode="rb", **_kwargs):
        return _zstd.ZstdDecompressor().stream_reader(infile)

    try:
        _fc.register_compression("zstd", _zstd_reader, "zst", force=True)
    except Exception:
        _fc.compr["zstd"] = _zstd_reader


_register_zstd_with_fsspec()

_OFFLINE_ENV_VARS = ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "TRANSFORMERS_OFFLINE")

# (builder_name, file suffixes it can load). Ordered by preference when a cache
# contains multiple formats.
_BUILDER_SUFFIXES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("parquet", (".parquet",)),
    ("json", (".jsonl", ".jsonl.gz", ".jsonl.zst", ".json.gz", ".json")),
    ("csv", (".csv", ".csv.gz")),
    ("text", (".txt", ".txt.gz")),
)


def hf_offline_mode() -> bool:
    """True if any HF offline env var is set to a truthy value."""
    return any(
        os.environ.get(var, "").strip().lower() in ("1", "true", "yes")
        for var in _OFFLINE_ENV_VARS
    )


def _default_datasets_cache() -> Path:
    explicit = os.environ.get("HF_DATASETS_CACHE")
    if explicit:
        return Path(explicit)
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        return Path(hf_home) / "datasets"
    return Path.home() / ".cache" / "huggingface" / "datasets"


def _has_suffix(name: str, suffixes: tuple[str, ...]) -> bool:
    lowered = name.lower()
    return any(lowered.endswith(s) for s in suffixes)


def resolve_cached_data_files(
    dataset_path: str, split: str = "train", config: str | None = None
) -> tuple[str, list[str]] | None:
    """
    Locate cached data shards for a `org/name` dataset + split inside the HF
    hub cache, and pick the datasets builder that can load them. Returns
    `(builder_name, sorted_file_paths)` or None if the cache has no usable
    shards, we are not offline, or the dataset_path looks like a local path.

    When *config* is given (e.g. a language pair like ``"en-fr"``), only files
    whose path contains that config name are considered.

    Supported layouts: parquet (preferred), jsonl / jsonl.zst / jsonl.gz,
    csv, plain text. Files under a `<split>/` subdir or named `<split>-*` /
    `<split>.<ext>` are treated as that split; otherwise all files of the
    chosen format are used.
    """
    if not hf_offline_mode():
        return None
    if not dataset_path or "/" not in dataset_path:
        return None
    if os.path.isabs(dataset_path) or os.path.isdir(dataset_path):
        return None

    org, name = dataset_path.split("/", 1)
    repo_dir = _default_datasets_cache() / f"datasets--{org}--{name}"
    refs_file = repo_dir / "refs" / "main"
    if not refs_file.is_file():
        return None
    rev = refs_file.read_text().strip()
    snap_dir = repo_dir / "snapshots" / rev
    if not snap_dir.is_dir():
        return None

    def is_config_file(p: Path) -> bool:
        """If a config is specified, only match files under that config subdir."""
        if config is None:
            return True
        # Check if any parent directory matches the config name
        return config in p.parts or config.replace("-", "_") in p.parts

    def is_split_file(p: Path, ext_match: str) -> bool:
        n = p.name
        parts = {x.lower() for x in p.parts}
        stem_no_ext = n[: -len(ext_match)] if n.lower().endswith(ext_match) else n
        return (
            stem_no_ext.startswith(f"{split}-")
            or stem_no_ext == split
            or split in parts
        )

    for builder, suffixes in _BUILDER_SUFFIXES:
        matches = [
            p for p in snap_dir.rglob("*")
            if p.is_file() and _has_suffix(p.name, suffixes) and is_config_file(p)
        ]
        if not matches:
            continue
        # Match to longest-suffix for split-file detection, so e.g. ".jsonl.zst"
        # beats ".zst".
        longest = sorted(suffixes, key=len, reverse=True)
        filtered: list[Path] = []
        for p in matches:
            low = p.name.lower()
            ext = next((s for s in longest if low.endswith(s)), "")
            if is_split_file(p, ext):
                filtered.append(p)
        chosen = filtered or matches
        return builder, sorted(str(p) for p in chosen)

    return None


def resolve_cached_parquet(
    dataset_path: str, split: str = "train"
) -> list[str] | None:
    """Deprecated alias kept for backwards compatibility. Returns the file
    list only when the cached format is parquet."""
    res = resolve_cached_data_files(dataset_path, split=split)
    if res is None:
        return None
    builder, files = res
    return files if builder == "parquet" else None


def offline_aware_load_dataset(
    dataset_path: str,
    name: str | None = None,
    split: str = "train",
    streaming: bool = True,
    **kwargs: Any,
):
    """
    Drop-in replacement for ``load_dataset(dataset_path, name, split=..., streaming=..., ...)``.

    *name* is the dataset config (e.g. a language pair like ``"en-fr"``).

    When offline and the cache has data shards for ``dataset_path`` (and
    optionally *name*), load them via the appropriate builder (no Hub call).
    Otherwise defer to vanilla ``load_dataset``.  ``trust_remote_code`` and
    other kwargs are forwarded to the fallback path.
    """
    resolved = resolve_cached_data_files(dataset_path, split=split, config=name)
    if resolved is not None:
        builder, files = resolved
        logger.info(
            f"[offline] loading {dataset_path!r} name={name!r} split={split!r} via "
            f"{builder!r} builder from {len(files)} cached shards under "
            f"{Path(files[0]).parent}"
        )
        return load_dataset(
            builder, data_files=files, split="train", streaming=streaming
        )
    return load_dataset(
        dataset_path, name, split=split, streaming=streaming, **kwargs
    )
