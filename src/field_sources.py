"""Pinned sparse presence and bounded, picklable prepared field inputs."""

import hashlib
import itertools
import json
import math
import os
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _digest(value: object) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError("expected a lowercase SHA256 hex hash")
    return value


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key: " + key)
        result[key] = value
    return result


def _invalid_constant(value: str) -> None:
    raise ValueError("nonfinite JSON constant: " + value)


def _json(raw: bytes) -> dict[str, Any]:
    result = json.loads(raw, object_pairs_hook=_unique_object, parse_constant=_invalid_constant)
    if not isinstance(result, dict):
        raise ValueError("expected a JSON object")
    return result


def _triple(value: Any, name: str) -> tuple[int, ...]:
    array = np.asarray(value)
    if array.shape != (3,) or array.dtype.kind not in "iu" or any(isinstance(v, (bool, np.bool_)) for v in value):
        raise ValueError(name + " must contain three integers in ZYX order")
    return tuple(int(v) for v in array)


def _bounds(lo: ArrayLike, hi: ArrayLike, shape: tuple[int, ...]) -> tuple[tuple[int, ...], tuple[int, ...]]:
    low, high = _triple(lo, "lower bounds"), _triple(hi, "upper bounds")
    if any(a < 0 or b <= a or b > n for a, b, n in zip(low, high, shape)):
        raise ValueError("bounds must be nonempty half-open ZYX intervals inside the source")
    return low, high


class CertifiedPresence:
    """Read a pinned, complete-coverage Zarr v2 uint8 inventory.

    ``manifest`` is a JSON file whose exact bytes match ``manifest_sha256``.
    Its fields are ``schema='winding-presence-v1'``, ``metadata_sha256``
    (the exact .zarray bytes), ``shape_zyx`` (equal to .zarray's shape),
    ``coverage_zyx`` and ``chunks`` (canonical relative ``z/y/x`` -> SHA256).
    Coverage is a nonempty list of boxes ``[[loZ, loY, loX], [hiZ, hiY, hiX]]``.
    Bounds are integer, half-open and in-volume; lows are chunk-aligned and
    highs are chunk-aligned or the final volume edge. Each box certifies every
    intersecting chunk completely, including clipped final chunks. Overlapping
    boxes are allowed. Uncovered chunks remain unknown, even if listed/present.

    Metadata requires C order, '/' separators, zero fill, no filters and chunks
    of at most 16 MiB decoded. Stored edge chunks have the full padded chunk
    shape. Compression uses a lazy optional ``numcodecs`` import (data extra).
    Reads recheck metadata and each touched chunk; missing listed payloads,
    symlinks, changed hashes and present unlisted payloads fail. Only absent,
    unlisted chunks within certified coverage are zero. Hashing and decoding
    consume the same bytes. No whole-volume inventory scan is performed.

    Receipts map absolute manifest/metadata/chunk paths to SHA256 strings.
    Implicit-zero hashes bind the pinned manifest, metadata and chunk key.
    The loaded manifest is an owned snapshot; its authority is its pinned hash.
    """

    def __init__(self, root: str | Path, manifest: str | Path, manifest_sha256: str):
        self.root = Path(root).resolve(strict=True)
        self._manifest_path = Path(manifest).resolve(strict=True)
        self._manifest_sha = _digest(manifest_sha256)
        raw = self._manifest_path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != self._manifest_sha:
            raise ValueError("presence manifest hash differs")
        document = _json(raw)
        if (
            set(document) != {"schema", "metadata_sha256", "shape_zyx", "coverage_zyx", "chunks"}
            or document.get("schema") != "winding-presence-v1"
        ):
            raise ValueError("invalid presence manifest schema")
        self._metadata_sha = _digest(document["metadata_sha256"])
        metadata = _json(self._metadata_bytes())
        self.shape = _triple(document["shape_zyx"], "shape_zyx")
        self._chunks = _triple(metadata.get("chunks"), "chunk shape")
        if (
            self.shape != _triple(metadata.get("shape"), "metadata shape")
            or any(n <= 0 for n in self.shape + self._chunks)
            or math.prod(self._chunks) > 16 * 1024**2
            or metadata.get("zarr_format") != 2
            or type(metadata.get("zarr_format")) is not int
            or metadata.get("dtype") != "|u1"
            or metadata.get("fill_value") != 0
            or isinstance(metadata.get("fill_value"), bool)
            or metadata.get("order") != "C"
            or metadata.get("dimension_separator") != "/"
            or "filters" not in metadata
            or metadata.get("filters") not in (None, [])
            or "compressor" not in metadata
        ):
            raise ValueError("invalid presence metadata or shape")
        self._compressor = metadata["compressor"]
        # Object codecs can execute pickle payloads; this reader accepts byte compression only.
        if self._compressor is not None and (
            not isinstance(self._compressor, dict)
            or not isinstance(self._compressor.get("id"), str)
            or self._compressor["id"] not in {"blosc", "zlib", "gzip", "bz2", "lz4", "zstd", "lzma"}
        ):
            raise ValueError("invalid presence compressor metadata")
        coverage = document["coverage_zyx"]
        if not isinstance(coverage, list) or not coverage:
            raise ValueError("complete chunk coverage boxes required")
        self._coverage = []
        for box in coverage:
            if not isinstance(box, list) or len(box) != 2:
                raise ValueError("coverage boxes must be [loZYX, hiZYX]")
            low, high = _bounds(box[0], box[1], self.shape)
            if any(a % c or (b % c and b != n) for a, b, c, n in zip(low, high, self._chunks, self.shape)):
                raise ValueError("coverage must certify complete chunk-aligned boxes")
            self._coverage.append((low, high))
        if not isinstance(document["chunks"], dict):
            raise ValueError("chunk inventory must be an object")
        self._inventory: dict[str, str] = {}
        for key, digest in document["chunks"].items():
            if re.fullmatch(r"(?:0|[1-9][0-9]*)/(?:0|[1-9][0-9]*)/(?:0|[1-9][0-9]*)", key) is None:
                raise ValueError("unsafe or noncanonical chunk path: " + key)
            self._require_coverage(tuple(int(v) for v in key.split("/")))
            self._inventory[key] = _digest(digest)

    def _safe_path(self, key: str) -> Path:
        path = self.root
        if path.resolve() != path or not path.is_dir():
            raise ValueError("presence root path changed or is not a directory")
        for part in key.split("/"):
            if path.exists() and not path.is_dir():
                raise ValueError("non-directory in presence path: " + str(path))
            path = path / part
            if path.is_symlink():
                raise ValueError("symlink in presence path: " + str(path))
        return path

    def _metadata_bytes(self) -> bytes:
        raw = self._safe_path(".zarray").read_bytes()
        if hashlib.sha256(raw).hexdigest() != self._metadata_sha:
            raise ValueError("presence metadata hash differs or metadata changed")
        return raw

    def _require_coverage(self, index: tuple[int, ...]) -> None:
        low = tuple(i * c for i, c in zip(index, self._chunks))
        high = tuple(min(a + c, n) for a, c, n in zip(low, self._chunks, self.shape))
        if any(a >= n for a, n in zip(low, self.shape)) or not any(
            all(a >= lo and b <= hi for a, b, lo, hi in zip(low, high, first, last)) for first, last in self._coverage
        ):
            raise ValueError("read or inventory outside certified complete chunk coverage")

    def read(self, lo: ArrayLike, hi: ArrayLike) -> tuple[NDArray[np.uint8], dict[str, str]]:
        low, high = _bounds(lo, hi, self.shape)
        self._metadata_bytes()
        receipts = {str(self._manifest_path): self._manifest_sha, str(self.root / ".zarray"): self._metadata_sha}
        result = np.empty(tuple(b - a for a, b in zip(low, high)), np.uint8)
        codec = None
        if self._compressor is not None:
            from numcodecs import get_codec

            codec = get_codec(self._compressor)
        for index in itertools.product(*(range(a // c, (b - 1) // c + 1) for a, b, c in zip(low, high, self._chunks))):
            self._require_coverage(index)
            key = "/".join(map(str, index))
            path = self._safe_path(key)
            origin = tuple(i * c for i, c in zip(index, self._chunks))
            first = tuple(max(a, o) for a, o in zip(low, origin))
            last = tuple(min(b, o + c) for b, o, c in zip(high, origin, self._chunks))
            target = tuple(slice(a - lo, b - lo) for a, b, lo in zip(first, last, low))
            if key not in self._inventory:
                if path.exists():
                    raise ValueError("unmanifested presence chunk appeared: " + str(path))
                result[target] = 0
                identity = f"Zarr-v2-implicit-zero:{self._manifest_sha}:{self._metadata_sha}:{key}"
                receipts[str(path)] = hashlib.sha256(identity.encode()).hexdigest()
                continue
            raw = path.read_bytes()
            digest = hashlib.sha256(raw).hexdigest()
            if digest != self._inventory[key]:
                raise ValueError("presence chunk hash differs: " + str(path))
            decoded = codec.decode(raw) if codec is not None else raw
            block = np.frombuffer(decoded, np.uint8)
            if block.size != math.prod(self._chunks):
                raise ValueError("decoded presence chunk size differs")
            source = tuple(slice(a - o, b - o) for a, b, o in zip(first, last, origin))
            result[target] = block.reshape(self._chunks)[source]
            receipts[str(path)] = digest
        return result, receipts


class NpyObservations:
    """Picklable bounded-crop loader of caller-owned immutable NPY files.

    ``hashes`` has exactly the keys target/weights/observed, each a full-file
    SHA256. Every open (including unpickling in a child) verifies those bytes.
    ``shape`` is ZYX; ``identity`` is a copy of the pinned source hash mapping.
    Targets are floating ZYX3, weights floating ZYX (float32 or float64), and
    observed is boolean ZYX. Arrays remain read-only memmaps; only requested
    crops are copied/validated. Fortran-ordered and non-native-endian floating
    sources are accepted. Returns owned C-contiguous float32/float32/bool arrays
    and SHA256 receipts, with finite targets and strictly positive finite
    weights at every voxel, including unobserved ones.

    Source receipt keys are absolute paths; crop keys append integer ZYX
    bounds. Each call hashes the copied *source-dtype* crop bytes before casting,
    so repeated changed crops produce changed receipts even if timestamps or
    float32 outputs match. The runner enforces receipt immutability. Replacement,
    truncation and header changes fail. Concurrent writers are unsupported;
    neither cached timestamps nor full-volume value validation are used.
    Pickling contains only paths/hashes, never memory-mapped array contents.
    """

    def __init__(
        self, target_path: str | Path, weights_path: str | Path, observed_path: str | Path, hashes: dict[str, str]
    ):
        names = ("target", "weights", "observed")
        if not isinstance(hashes, dict) or set(hashes) != set(names):
            raise ValueError("hashes must contain exactly target, weights, observed")
        self._hashes = {key: _digest(hashes[key]) for key in names}
        self._paths = dict(zip(names, (Path(p).absolute() for p in (target_path, weights_path, observed_path))))
        self._arrays: dict[str, np.memmap] = {}
        self._files: dict[str, tuple[int, int, int]] = {}
        self._headers: dict[str, str] = {}
        for key, path in self._paths.items():
            with path.open("rb") as stream:
                stat = os.fstat(stream.fileno())
                if hashlib.file_digest(stream, "sha256").hexdigest() != self._hashes[key]:
                    raise ValueError("NPY source hash differs: " + key)
                array = np.load(path, mmap_mode="r", allow_pickle=False)
                if not isinstance(array, np.memmap):
                    if isinstance(array, np.lib.npyio.NpzFile):
                        array.close()
                    raise ValueError("expected a memory-mappable NPY array")
                self._arrays[key] = array
                self._files[key] = (stat.st_dev, stat.st_ino, stat.st_size)
                stream.seek(0)
                self._headers[key] = hashlib.sha256(stream.read(array.offset)).hexdigest()
            self._check_file(key)
        target, weights, observed = (self._arrays[key] for key in names)
        if target.ndim != 4 or target.shape[-1] != 3 or any(n <= 0 for n in target.shape):
            raise ValueError("target shape must be nonempty ZYX3")
        self.shape = tuple(int(v) for v in target.shape[:3])
        if weights.shape != self.shape or observed.shape != self.shape:
            raise ValueError("weights and observed shape must match target ZYX")
        if (
            any(a.dtype.kind != "f" or a.dtype.itemsize not in (4, 8) for a in (target, weights))
            or observed.dtype != np.bool_
        ):
            raise ValueError("expected float32/float64 target/weights and boolean observed dtype")

    @property
    def identity(self) -> dict[str, str]:
        return self._hashes.copy()

    def __reduce__(self) -> tuple[Any, tuple[Any, ...]]:
        return type(self), (*self._paths.values(), self._hashes)

    def _check_file(self, key: str) -> None:
        with self._paths[key].open("rb") as stream:
            stat = os.fstat(stream.fileno())
            if (stat.st_dev, stat.st_ino, stat.st_size) != self._files[key] or hashlib.sha256(
                stream.read(self._arrays[key].offset)
            ).hexdigest() != self._headers[key]:
                raise ValueError("NPY source file or header changed: " + key)

    def __call__(
        self, lo_zyx: ArrayLike, hi_zyx: ArrayLike
    ) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.bool_], dict[str, str]]:
        low, high = _bounds(lo_zyx, hi_zyx, self.shape)
        region = tuple(slice(a, b) for a, b in zip(low, high))
        bounds_key = ",".join(map(str, low)) + ":" + ",".join(map(str, high))
        receipts = {}
        crops: list[NDArray[Any]] = []
        for key, path in self._paths.items():
            self._check_file(key)
            crop = np.array(self._arrays[key][region], order="C", copy=True)
            receipts[str(path)] = self._hashes[key]
            receipts[f"{path}#crop:{bounds_key}"] = hashlib.sha256(crop.tobytes()).hexdigest()
            crops.append(crop)
        with np.errstate(over="ignore", under="ignore"):
            target = crops[0].astype(np.float32, copy=False)
            weights = crops[1].astype(np.float32, copy=False)
        observed = crops[2].astype(np.bool_, copy=False)
        if not np.isfinite(target).all():
            raise ValueError("target crop must be finite float32")
        if not np.isfinite(weights).all() or not (weights > 0).all():
            raise ValueError("weights crop must be finite and strictly positive float32")
        return target, weights, observed, receipts


class NpyFixedValues:
    """Half-open ZYX crops of a pinned, caller-owned immutable fixed-value NPY.

    The positive integer ``shape`` must match the float32/float64 source. NaN
    marks free nodes; finite values, including exact zero, are fixed. Crops are
    owned C-contiguous arrays retaining source precision and byte order; only
    the requested crop is checked for infinities. Fortran sources are accepted.

    Every open verifies the full-file SHA256 without a resident volume copy.
    Calls recheck file identity/size and header, as for ``NpyObservations``;
    concurrent/in-place writers are unsupported. ``identity`` is a detached
    mapping with absolute ``path`` and pinned ``sha256``. Pickling stores only
    path/hash/shape and reopens/verifies the source, never the memmap contents.
    """

    def __init__(self, path: str | Path, sha256: str, shape: Sequence[int]) -> None:
        self._path = Path(path).absolute()
        self._sha256 = _digest(sha256)
        self.shape = _triple(shape, "fixed shape")
        if any(n <= 0 for n in self.shape):
            raise ValueError("fixed shape must be positive ZYX")
        with self._path.open("rb") as stream:
            stat = os.fstat(stream.fileno())
            if hashlib.file_digest(stream, "sha256").hexdigest() != self._sha256:
                raise ValueError("fixed NPY source hash differs")
            array = np.load(self._path, mmap_mode="r", allow_pickle=False)
            if not isinstance(array, np.memmap):
                if isinstance(array, np.lib.npyio.NpzFile):
                    array.close()
                raise ValueError("expected a memory-mappable fixed NPY array")
            if array.shape != self.shape:
                raise ValueError("fixed NPY source shape differs")
            if array.dtype.kind != "f" or array.dtype.itemsize not in (4, 8):
                raise ValueError("fixed NPY dtype must be float32/float64")
            self._array = array
            self._file = (stat.st_dev, stat.st_ino, stat.st_size)
            stream.seek(0)
            self._header = hashlib.sha256(stream.read(array.offset)).hexdigest()
        self._check_file()

    @property
    def identity(self) -> dict[str, str]:
        return {"path": str(self._path), "sha256": self._sha256}

    def __reduce__(self) -> tuple[Any, tuple[Any, ...]]:
        return type(self), (self._path, self._sha256, self.shape)

    def _check_file(self) -> None:
        with self._path.open("rb") as stream:
            stat = os.fstat(stream.fileno())
            if (stat.st_dev, stat.st_ino, stat.st_size) != self._file or hashlib.sha256(
                stream.read(self._array.offset)
            ).hexdigest() != self._header:
                raise ValueError("fixed NPY source file or header changed")

    def __call__(self, lo: ArrayLike, hi: ArrayLike) -> NDArray[np.float32] | NDArray[np.float64]:
        low, high = _bounds(lo, hi, self.shape)
        self._check_file()
        crop = np.array(self._array[tuple(slice(a, b) for a, b in zip(low, high))], order="C", copy=True)
        if np.isinf(crop).any():
            raise ValueError("fixed crop must contain only finite values or NaN free markers")
        return crop
