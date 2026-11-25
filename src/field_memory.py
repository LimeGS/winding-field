"""Host-memory plan for block field solves: how much of the run fits in RAM.

Three tiers, chosen from the memory available when the run starts:

- ``disk``: the field does not fit. Chunks are read from and committed to disk block by
  block, and the plan carries a warning.
- ``field``: the decoded field fits. Every chunk stays cached, commits are published once
  per sweep in the background (``solve_blocks(checkpoint="sweep")``), and what is left
  over caches observation boxes until it runs out.
- ``all``: the field and every observation box fit; after the first sweep no block
  reads its inputs from disk again.

Sizes are logical array bytes with per-entry overheads, not allocator RSS, so the plan
keeps a reserve: ``max(reserve_min_bytes, reserve_fraction * available)`` plus
``workspace_bytes`` for the caller's own solver memory.
"""

from __future__ import annotations

import math
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

__all__ = ["available_memory", "field_bytes", "observation_bytes", "plan_memory"]

TIERS = ("disk", "field", "all")
_CHUNK_OVERHEAD = 512  # BlockStore._charge per retained chunk
_BOX_OVERHEAD = 4096  # CLI observation cache charge per retained box
# Pending sweep commits are held next to the cached chunks while a flush writes them.
_WRITE_BACK_FACTOR = 1.25


def _read_int(path: Path) -> int | None:
    try:
        text = path.read_text().strip()
    except OSError:
        return None
    return int(text) if text.isdigit() else None


def available_memory() -> tuple[int | None, str]:
    """Bytes this process can still allocate without swapping, and where that came from.

    Linux: ``MemAvailable`` from ``/proc/meminfo``, lowered by a cgroup v2 or v1 limit.
    Elsewhere: free physical pages from ``sysconf``. ``(None, "unknown")`` if neither works.
    """
    found: list[tuple[int, str]] = []
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                found.append((int(line.split()[1]) * 1024, "MemAvailable"))
                break
    except (OSError, ValueError, IndexError):
        pass
    cgroup = Path("/sys/fs/cgroup")
    for limit_path, usage_path in (
        (cgroup / "memory.max", cgroup / "memory.current"),
        (cgroup / "memory/memory.limit_in_bytes", cgroup / "memory/memory.usage_in_bytes"),
    ):
        limit, usage = _read_int(limit_path), _read_int(usage_path)
        # cgroup v1 reports "no limit" as a huge page-aligned number.
        if limit is not None and usage is not None and limit < 1 << 60:
            found.append((max(limit - usage, 0), "cgroup"))
            break
    if not found:
        try:
            pages = os.sysconf("SC_AVPHYS_PAGES")
            size = os.sysconf("SC_PAGE_SIZE")
            if pages > 0 and size > 0:
                found.append((pages * size, "sysconf"))
        except (OSError, ValueError, AttributeError):
            pass
    if not found:
        return None, "unknown"
    return min(found)


def _shape(shape: Sequence[int]) -> tuple[int, int, int]:
    out = tuple(int(n) for n in shape)
    if len(out) != 3 or min(out) < 1:
        raise ValueError("shape must be three positive integers")
    return out  # type: ignore[return-value]


def field_bytes(shape: Sequence[int], chunk: int) -> int:
    """Decoded field held by ``BlockStore``: float32 deltas, packed support, per-chunk overhead."""
    shape = _shape(shape)
    nodes = math.prod(shape)
    chunks = math.prod(-(-n // int(chunk)) for n in shape)
    return 4 * nodes + -(-nodes // 8) + chunks * _CHUNK_OVERHEAD


def observation_bytes(shape: Sequence[int], chunk: int, halo: int, bytes_per_node: int = 17) -> int:
    """Every block's observation crop (core plus halo): target xyz, weight and support by default."""
    shape = _shape(shape)
    chunk, halo = int(chunk), int(halo)
    total = 1
    for n in shape:
        # Crops are separable: the summed crop volume is the product of per-axis sums.
        total *= sum(min(k + chunk + halo, n) - max(k - halo, 0) for k in range(0, n, chunk))
    boxes = math.prod(-(-n // chunk) for n in shape)
    return bytes_per_node * total + _BOX_OVERHEAD * boxes


def plan_memory(
    shape: Sequence[int],
    chunk: int,
    halo: int,
    *,
    tier: str = "auto",
    available: int | None = None,
    workspace_bytes: int = 0,
    reserve_fraction: float = 0.15,
    reserve_min_bytes: int = 4 << 30,
    observation_bytes_per_node: int = 17,
) -> dict[str, Any]:
    """Pick the memory tier (``tier="auto"``) or apply a forced one, with its budgets.

    Returns JSON with ``tier``, ``requested``, ``available_bytes`` and its ``source``,
    ``reserve_bytes``, ``usable_bytes``, ``field_bytes``, ``observation_bytes``,
    ``field_cache_bytes``, ``observation_cache_bytes``, ``checkpoint`` (``block`` or
    ``sweep``), ``observations_cached_fraction`` and ``warning`` (``None`` when nothing
    is wrong). A forced tier is applied even if it does not fit; the warning says so.
    """
    if tier not in ("auto", *TIERS):
        raise ValueError("memory tier must be auto, disk, field or all")
    if not 0 <= reserve_fraction < 1:
        raise ValueError("reserve_fraction must be in [0, 1)")
    source = "given"
    if available is None:
        available, source = available_memory()
    field = field_bytes(shape, chunk)
    observations = observation_bytes(shape, chunk, halo, observation_bytes_per_node)
    resident = math.ceil(field * _WRITE_BACK_FACTOR)
    if available is None:
        reserve = usable = 0
    else:
        reserve = max(int(reserve_min_bytes), int(reserve_fraction * available)) + int(workspace_bytes)
        usable = max(int(available) - reserve, 0)
    fits_field = available is not None and resident <= usable
    fits_all = available is not None and resident + observations <= usable
    chosen = tier if tier != "auto" else ("all" if fits_all else "field" if fits_field else "disk")
    gib = 1 << 30
    warning = None
    if available is None:
        warning = "host memory could not be measured"
        if tier == "auto":
            warning += "; the field stays on disk"
    elif chosen == "disk" and tier == "auto":
        warning = (
            f"not enough RAM for the field ({resident / gib:.1f} GiB needed, {usable / gib:.1f} GiB usable "
            f"of {available / gib:.1f} GiB available): reading and writing every block on disk"
        )
    elif (chosen == "field" and not fits_field) or (chosen == "all" and not fits_all):
        need = resident + (observations if chosen == "all" else 0)
        warning = f"forced tier {chosen} needs {need / gib:.1f} GiB but only {usable / gib:.1f} GiB are usable"
    field_cache = observation_cache = 0
    if chosen in ("field", "all"):
        field_cache = field
        observation_cache = observations if chosen == "all" else max(usable - resident, 0)
    return dict(
        tier=chosen,
        requested=tier,
        available_bytes=available,
        source=source,
        reserve_bytes=reserve,
        usable_bytes=usable,
        field_bytes=field,
        observation_bytes=observations,
        field_cache_bytes=field_cache,
        observation_cache_bytes=observation_cache,
        checkpoint="block" if chosen == "disk" else "sweep",
        observations_cached_fraction=min(observation_cache / observations, 1.0) if observations else 1.0,
        warning=warning,
    )
