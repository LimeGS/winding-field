"""Tiny real checkpoints: numerical oracles, ownership, and interrupted publication."""

import hashlib
import importlib
import io
import itertools
import json
import os
import stat
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch
from uuid import UUID
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import pytest

IDENTITY = {"source_sha256": "a" * 64, "spacing_xyz": [1.0, 2.0, 3.0]}
PROGRESS = {"sweep": 2, "next_block": 3, "order_identity": "b" * 64, "sweep_max_update": 17.5, "sum": 23.0}


def api():
    return importlib.import_module("field_store")


def create(root, shape=(2, 2, 2), chunk=2, **kwargs):
    return api().BlockStore(root, shape, chunk, IDENTITY, **kwargs)


def checkpoint(root):
    return json.loads((root / "CHECKPOINT.json").read_bytes())


def publish_manifest(root, manifest):
    (root / "CHECKPOINT.json").write_text(json.dumps(manifest))


def rewrite_payload(root, raw):
    manifest = checkpoint(root)
    record = manifest["chunks"]["0-0-0"]
    (root / record["path"]).write_bytes(raw)
    record.update(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
    publish_manifest(root, manifest)


def npz_bytes(**arrays):
    stream = io.BytesIO()
    np.savez_compressed(stream, **arrays)
    return stream.getvalue()


def seed(root, **kwargs):
    with create(root, **kwargs) as store:
        store.commit((0, 0, 0), np.full((2, 2, 2), 7.0), PROGRESS, np.ones((2, 2, 2), bool))
    return checkpoint(root)


def test_default_zero_is_unobserved_and_context_releases_writer(tmp_path):
    root = tmp_path / "field"
    with create(root) as store:
        assert store.root == root
        assert store.shape == (2, 2, 2)
        assert store.chunk == 2
        value = store.read((0, 0, 0), (2, 2, 2))
        assert value.dtype == np.float64
        np.testing.assert_array_equal(value, np.zeros((2, 2, 2)))
        observed = store.read_observed((0, 0, 0), (2, 2, 2))
        assert observed.dtype == np.bool_
        assert not observed.any()
        assert store.cache_nbytes == 0
        assert store.stats["zeros"] >= 1
    store.close()
    with pytest.raises(RuntimeError):
        store.read((0, 0, 0), (2, 2, 2))
    with create(root, resume=True) as reopened:
        assert reopened.manifest["progress"] == {}


@pytest.mark.parametrize("workers", [1, 3])
def test_edges_offsets_grouped_nodes_and_resume(tmp_path, workers):
    root = tmp_path / "field"
    shape = (3, 3, 5)
    z, y, x = np.indices(shape)
    expected = 1e12 + 100 * z + 10 * y + x
    support = (z + y + x) % 3 == 1
    with create(root, shape=shape, workers=workers) as store:
        for key in itertools.product(range(2), range(2), range(3)):
            slices = tuple(slice(k * 2, min(k * 2 + 2, n)) for k, n in zip(key, shape))
            store.commit(key, expected[slices], PROGRESS, support[slices])
        np.testing.assert_array_equal(store.read((1, 1, 1), shape), expected[1:, 1:, 1:])
        np.testing.assert_array_equal(store.read_observed((0, 0, 0), shape), support)
        nodes = np.array([[44, 0, 44], [17, 5, 1]])
        np.testing.assert_array_equal(
            store.nodes(nodes), [[1e12 + 224, 1e12, 1e12 + 224], [1e12 + 102, 1e12 + 10, 1e12 + 1]]
        )
        assert store.nodes(np.empty((0, 2), dtype=int)).shape == (0, 2)
        assert store.nodes([]).size == 0
        assert store.nodes(np.array(44, dtype=np.uint64)) == 1e12 + 224
        assert store.cache_nbytes == 0
    with create(root, shape=shape, workers=workers, resume=True) as store:
        np.testing.assert_array_equal(store.read((0, 0, 0), shape), expected)
        assert store.manifest["progress"] == PROGRESS


def test_observed_zero_and_quantization_receipt_survive_restart(tmp_path):
    root = tmp_path / "field"
    observed = np.array([True, False, True, False, False, False, False, True]).reshape(2, 2, 2)
    with create(root) as store:
        zero_receipt = store.commit((0, 0, 0), np.zeros((2, 2, 2)), PROGRESS, observed)
        assert zero_receipt["quantization_max"] == 0
        assert zero_receipt["stored_bytes"] > 0
    with create(root, resume=True) as store:
        np.testing.assert_array_equal(store.read_observed((0, 0, 0), (2, 2, 2)), observed)
        values = np.array([0, np.pi, 1, -1, 2, -2, 1 / 3, -1 / 3]).reshape(2, 2, 2)
        result = store.commit((0, 0, 0), values, PROGRESS)
        actual = store.read((0, 0, 0), (2, 2, 2))
        actual_error = float(np.max(np.abs(actual - values)))
        assert 0 < actual_error < 1e-6
        assert result["quantization_max"] == actual_error
        assert result["stored_bytes"] == sum(p.stat().st_size for p in root.glob("*.npz"))
        assert not store.read_observed((0, 0, 0), (2, 2, 2)).any()
        store.commit((0, 0, 0), np.zeros((2, 2, 2)), {"next_block": 4})
        assert store.manifest["chunks"] == {}
        assert not list(root.glob("*.npz"))
    with create(root, resume=True) as store:
        assert not store.read_observed((0, 0, 0), (2, 2, 2)).any()
        assert store.manifest["progress"] == {"next_block": 4}


def test_cache_is_bounded_lru_and_commits_invalidate_generations(tmp_path):
    root = tmp_path / "field"
    with create(root, shape=(2, 2, 6), cache_bytes=600) as store:
        for x in range(3):
            store.commit((0, 0, x), np.full((2, 2, 2), x + 1.0), {}, np.ones((2, 2, 2), bool))
        for x in (0, 0, 1, 2, 0):
            np.testing.assert_array_equal(store.read((0, 0, 2 * x), (2, 2, 2 * x + 2)), x + 1.0)
            assert 0 < store.cache_nbytes <= 600
        assert store.stats["hits"] >= 1
        assert store.stats["evictions"] >= 2
        assert store.stats["peak_cache_bytes"] <= 600
        old = store.manifest["chunks"]["0-0-0"]["path"]
        store.commit((0, 0, 0), np.full((2, 2, 2), -5.0), {})
        assert not (root / old).exists()
        np.testing.assert_array_equal(store.read((0, 0, 0), (2, 2, 2)), -5.0)
        assert not store.read_observed((0, 0, 0), (2, 2, 2)).any()
    assert store.cache_nbytes == 0
    with create(root, shape=(2, 2, 6), resume=True, cache_bytes=1) as store:
        store.read((0, 0, 0), (2, 2, 2))
        store.read((0, 0, 0), (2, 2, 2))
        assert store.cache_nbytes == 0
        assert store.stats["hits"] == 0


@pytest.mark.parametrize(
    "option,value",
    [
        ("cache_bytes", -1),
        ("cache_bytes", True),
        ("workers", 0),
        ("workers", 1.5),
        ("max_bytes", -1),
        ("reserve_bytes", False),
        ("chunk", 0),
        ("chunk", 2.0),
        ("shape", (2, 0, 2)),
        ("shape", (2, 2)),
        ("shape", (2, True, 2)),
        ("shape", (2, 2.0, 2)),
        ("shape", (2**63, 2, 2)),
    ],
)
def test_invalid_configuration_creates_no_checkpoint(tmp_path, option, value):
    with pytest.raises(ValueError):
        create(tmp_path / "field", **{option: value})
    assert not (tmp_path / "field" / "CHECKPOINT.json").exists()


@pytest.mark.parametrize(
    "lo,hi",
    [
        ((-1, 0, 0), (1, 2, 2)),
        ((0, 0, 0), (3, 2, 2)),
        ((0, 0, 0), (0, 2, 2)),
        ((0.0, 0, 0), (2, 2, 2)),
        ((False, 0, 0), (2, 2, 2)),
        ((0, 0), (2, 2)),
        ((0, 0, 0), (2, 2, np.nan)),
    ],
)
def test_bounds_are_strict_integer_half_open_boxes(tmp_path, lo, hi):
    with create(tmp_path / "field") as store:
        for method in (store.read, store.read_observed):
            with pytest.raises(ValueError):
                method(lo, hi)


@pytest.mark.parametrize(
    "nodes",
    [[-1], [8], [1.1], [True], [True, 1], [[0, 1], [2, False]], [np.nan], [1j], np.array([2**64 - 1], dtype=np.uint64)],
)
def test_node_indices_are_not_truncated_or_wrapped(tmp_path, nodes):
    with create(tmp_path / "field") as store:
        with pytest.raises(ValueError):
            store.nodes(nodes)


@pytest.mark.parametrize("key", [(0, 0, -1), (1, 0, 0), (0.0, 0, 0), (True, 0, 0), (0, 0), "000"])
def test_commit_key_must_identify_an_owned_chunk(tmp_path, key):
    with create(tmp_path / "field") as store:
        before = (store.root / "CHECKPOINT.json").read_bytes()
        with pytest.raises(ValueError):
            store.commit(key, np.ones((2, 2, 2)), {})
        assert (store.root / "CHECKPOINT.json").read_bytes() == before


@pytest.mark.parametrize(
    "values",
    [
        np.ones((1, 2, 2)),
        np.full((2, 2, 2), np.nan),
        np.full((2, 2, 2), np.inf),
        np.full((2, 2, 2), 1j),
        np.full((2, 2, 2), 1e308),
        np.array([1e40, -1e40] * 4).reshape(2, 2, 2),
    ],
)
def test_invalid_or_nonfinite_encoding_fails_before_publication(tmp_path, values):
    root = tmp_path / "field"
    seed(root)
    with create(root, resume=True) as store:
        before = {p.name: p.read_bytes() for p in root.iterdir() if p.is_file()}
        with pytest.raises(ValueError):
            store.commit((0, 0, 0), values, {})
        assert {p.name: p.read_bytes() for p in root.iterdir() if p.is_file()} == before
        np.testing.assert_array_equal(store.read((0, 0, 0), (2, 2, 2)), 7)


@pytest.mark.parametrize("observed", [np.ones((2, 2, 2), np.uint8), np.ones((2, 2), bool)])
def test_observation_mask_requires_exact_boolean_shape(tmp_path, observed):
    with create(tmp_path / "field") as store:
        with pytest.raises(ValueError):
            store.commit((0, 0, 0), np.zeros((2, 2, 2)), {}, observed)
        assert store.manifest["chunks"] == {}


def test_identity_progress_and_manifest_do_not_retain_mutable_aliases(tmp_path):
    root = tmp_path / "field"
    identity = {"source": [1, 2]}
    progress = {"sweep": 1, "stats": {"maximum": 42}}
    with api().BlockStore(root, (2, 2, 2), 2, identity) as store:
        identity["source"][0] = 999
        store.save_progress(progress)
        progress["stats"]["maximum"] = -1
        exposed = store.manifest
        exposed["chunks"]["bad"] = {}
        exposed["progress"]["stats"]["maximum"] = -2
        assert store.manifest["progress"]["stats"]["maximum"] == 42
    with api().BlockStore(root, (2, 2, 2), 2, {"source": [1, 2]}, resume=True) as store:
        assert store.manifest["progress"] == {"sweep": 1, "stats": {"maximum": 42}}
    for changed in ({"source": [2, 1]}, {"source": [True, 2]}, {"source": [1.0, 2]}):
        with pytest.raises(ValueError):
            api().BlockStore(root, (2, 2, 2), 2, changed, resume=True)


@pytest.mark.parametrize("progress", [[], {"maximum": np.nan}, {"not_json": {1, 2}}])
def test_invalid_progress_never_advances_field_or_cursor(tmp_path, progress):
    root = tmp_path / "field"
    seed(root)
    with create(root, resume=True) as store:
        before = checkpoint(root)
        for action in (
            lambda: store.save_progress(progress),
            lambda: store.commit((0, 0, 0), np.ones((2, 2, 2)), progress),
        ):
            with pytest.raises(ValueError):
                action()
            assert checkpoint(root) == before
        np.testing.assert_array_equal(store.read((0, 0, 0), (2, 2, 2)), 7)


def test_single_writer_and_failed_open_release_the_lease(tmp_path):
    root = tmp_path / "field"
    with create(root) as owner:
        with pytest.raises((RuntimeError, OSError)):
            create(root, resume=True)
        with pytest.raises(FileExistsError):
            create(root)
        owner.save_progress(PROGRESS)
    with pytest.raises(ValueError):
        create(root, shape=(3, 2, 2), resume=True)
    with create(root, resume=True) as reopened:
        assert reopened.manifest["progress"] == PROGRESS


@pytest.mark.parametrize(
    "change",
    [
        "identity",
        "shape",
        "chunk",
        "chunks",
        "progress",
        "label",
        "record",
        "path",
        "absolute",
        "backslash",
        "wrong_generation",
        "sha256",
        "bytes",
        "quantization",
        "missing",
    ],
)
def test_malformed_manifests_are_rejected_on_open(tmp_path, change):
    root = tmp_path / "field"
    manifest = seed(root)
    record = manifest["chunks"]["0-0-0"]
    if change in {"identity", "shape", "chunk", "chunks", "progress"}:
        manifest[change] = {
            "identity": {"changed": True},
            "shape": [2.0, 2, 2],
            "chunk": True,
            "chunks": [],
            "progress": [],
        }[change]
    elif change == "label":
        manifest["chunks"]["00-0-0"] = manifest["chunks"].pop("0-0-0")
    elif change == "record":
        manifest["chunks"]["0-0-0"] = []
    elif change in {"path", "absolute", "backslash", "wrong_generation"}:
        record["path"] = {
            "path": "../outside.npz",
            "absolute": "/outside.npz",
            "backslash": "..\\outside.npz",
            "wrong_generation": "1-0-0.g123.npz",
        }[change]
    elif change == "sha256":
        record["sha256"] = "x" * 64
    elif change == "bytes":
        record["bytes"] = True
    elif change == "quantization":
        record["quantization_max"] = -1
    else:
        del record["sha256"]
    publish_manifest(root, manifest)
    with pytest.raises(ValueError):
        create(root, resume=True)


@pytest.mark.parametrize(
    "raw", [b"{}", b"[]", b"not json", b'{"shape":[2,2,2],"shape":[2,2,2]}', b'{"n":NaN}', b'{"n":1e999}']
)
def test_noncanonical_or_nonfinite_manifest_json_is_rejected(tmp_path, raw):
    root = tmp_path / "field"
    seed(root)
    (root / "CHECKPOINT.json").write_bytes(raw)
    with pytest.raises(ValueError):
        create(root, resume=True)


@pytest.mark.parametrize("member", ["root", "manifest", "chunk", "lease"])
def test_symlinks_cannot_redirect_store_members(tmp_path, member):
    root = tmp_path / "field"
    manifest = seed(root)
    if member == "root":
        link = tmp_path / "alias"
        link.symlink_to(root, target_is_directory=True)
        with pytest.raises(ValueError):
            create(link, resume=True)
        return
    name = {"manifest": "CHECKPOINT.json", "chunk": manifest["chunks"]["0-0-0"]["path"], "lease": ".writer.lock"}[
        member
    ]
    target = tmp_path / "outside"
    original = (root / name).read_bytes()
    target.write_bytes(original)
    (root / name).unlink()
    (root / name).symlink_to(target)
    with pytest.raises((ValueError, OSError)):
        create(root, resume=True)
    assert target.read_bytes() == original


@pytest.mark.parametrize("corruption", ["hash", "size", "missing"])
def test_restore_validates_every_referenced_payload(tmp_path, corruption):
    root = tmp_path / "field"
    manifest = seed(root)
    path = root / manifest["chunks"]["0-0-0"]["path"]
    if corruption == "missing":
        path.unlink()
    elif corruption == "size":
        path.write_bytes(b"truncated")
    else:
        raw = bytearray(path.read_bytes())
        raw[len(raw) // 2] ^= 1
        path.write_bytes(raw)
    with pytest.raises((ValueError, OSError)):
        create(root, resume=True)


@pytest.mark.parametrize(
    "change",
    [
        "delta_dtype",
        "delta_shape",
        "delta_nan",
        "offset_dtype",
        "offset_shape",
        "offset_inf",
        "mask_dtype",
        "mask_size",
        "mask_padding",
        "extra",
        "missing",
        "not_zip",
        "huge_header",
    ],
)
def test_digest_does_not_excuse_malformed_npz_members(tmp_path, change):
    root = tmp_path / "field"
    with create(root, shape=(1, 1, 1)) as store:
        store.commit((0, 0, 0), np.ones((1, 1, 1)), {})
    arrays = {"delta": np.zeros((1, 1, 1), np.float32), "offset": np.float64(7), "observed": np.array([128], np.uint8)}
    modifications = {
        "delta_dtype": ("delta", np.zeros((1, 1, 1), np.float64)),
        "delta_shape": ("delta", np.zeros((1, 1, 2), np.float32)),
        "delta_nan": ("delta", np.full((1, 1, 1), np.nan, np.float32)),
        "offset_dtype": ("offset", np.float32(7)),
        "offset_shape": ("offset", np.array([7.0])),
        "offset_inf": ("offset", np.float64(np.inf)),
        "mask_dtype": ("observed", np.array([True])),
        "mask_size": ("observed", np.array([1, 2], np.uint8)),
        "mask_padding": ("observed", np.array([129], np.uint8)),
        "extra": ("other", np.ones(1)),
    }
    if change in modifications:
        name, value = modifications[change]
        arrays[name] = value
    if change == "missing":
        del arrays["observed"]
    raw = npz_bytes(**arrays)
    if change == "not_zip":
        raw = b"not an npz"
    if change == "huge_header":
        stream = io.BytesIO()
        with ZipFile(io.BytesIO(raw)) as source, ZipFile(stream, "w", ZIP_DEFLATED) as dest:
            for name in source.namelist():
                data = source.read(name)
                if name == "delta.npy":
                    header = io.BytesIO()
                    np.lib.format.write_array_header_1_0(
                        header, {"descr": "<f4", "fortran_order": False, "shape": (2**40,)}
                    )
                    data = header.getvalue()
                dest.writestr(name, data)
        raw = stream.getvalue()
    rewrite_payload(root, raw)
    with pytest.raises(ValueError):
        create(root, shape=(1, 1, 1), resume=True)


def test_hash_and_decode_use_one_opened_byte_snapshot(tmp_path, monkeypatch):
    root = tmp_path / "field"
    manifest = seed(root)
    path = root / manifest["chunks"]["0-0-0"]["path"]
    original_bytes = path.read_bytes()
    real_sha256 = hashlib.sha256
    replaced = False

    def replace_after_hash(data=b"", **kwargs):
        nonlocal replaced
        result = real_sha256(data, **kwargs)
        if data == original_bytes and not replaced:
            replaced = True
            path.write_bytes(
                npz_bytes(delta=np.zeros((2, 2, 2), np.float32), offset=np.float64(99), observed=np.empty(0, np.uint8))
            )
        return result

    with create(root, resume=True) as store:
        monkeypatch.setattr(hashlib, "sha256", replace_after_hash)
        np.testing.assert_array_equal(store.read((0, 0, 0), (2, 2, 2)), 7)
        assert replaced
        with pytest.raises(ValueError):
            store.read((0, 0, 0), (2, 2, 2))


@pytest.mark.parametrize("limit", ["max_bytes", "reserve_bytes"])
def test_budget_preserves_checkpoint_and_unreferenced_files(tmp_path, limit):
    root = tmp_path / "field"
    seed(root)
    orphan = root / "unreferenced.npz"
    orphan.write_bytes(b"do not remove")
    baseline = {p.name: p.read_bytes() for p in root.iterdir()}
    used = sum(len(b) for b in baseline.values())
    options = {limit: used + 1 if limit == "max_bytes" else 2**62}
    with create(root, resume=True, **options) as store:
        with pytest.raises(api().StorageBudgetError):
            store.commit((0, 0, 0), np.full((2, 2, 2), 9), {"next_block": 4})
        with pytest.raises(RuntimeError):
            store.save_progress({})
    assert {p.name: p.read_bytes() for p in root.iterdir()} == baseline
    with create(root, resume=True) as store:
        assert store.manifest["progress"] == PROGRESS
        np.testing.assert_array_equal(store.read((0, 0, 0), (2, 2, 2)), 7)


def test_cap_accounts_for_old_generation_and_both_replacement_temporaries(tmp_path):
    root = tmp_path / "field"
    manifest = seed(root)
    used = sum(p.stat().st_size for p in root.iterdir())
    payload_size = manifest["chunks"]["0-0-0"]["bytes"]
    with create(root, resume=True, max_bytes=used + payload_size + 10) as store:
        with pytest.raises(api().StorageBudgetError):
            store.commit((0, 0, 0), np.full((2, 2, 2), 9), PROGRESS, np.ones((2, 2, 2), bool))
    assert checkpoint(root) == manifest


@pytest.mark.parametrize(
    "stage",
    ["payload_fsync", "payload_rename", "payload_dir_fsync", "manifest_fsync", "manifest_rename", "manifest_dir_fsync"],
)
def test_interrupted_commit_reopens_exact_published_generation(tmp_path, monkeypatch, stage):
    root = tmp_path / "field"
    previous = seed(root)
    old_path = root / previous["chunks"]["0-0-0"]["path"]
    real_fsync, real_replace = os.fsync, os.replace
    file_syncs = directory_syncs = 0

    def fsync(fd):
        nonlocal file_syncs, directory_syncs
        if stat.S_ISDIR(os.fstat(fd).st_mode):
            directory_syncs += 1
            current = "payload_dir_fsync" if directory_syncs == 1 else "manifest_dir_fsync"
        else:
            file_syncs += 1
            current = "payload_fsync" if file_syncs == 1 else "manifest_fsync"
        if stage == current:
            raise OSError("injected fsync failure")
        return real_fsync(fd)

    def replace(source, destination, *args, **kwargs):
        current = "manifest_rename" if Path(destination).name == "CHECKPOINT.json" else "payload_rename"
        if stage == current:
            raise OSError("injected rename failure")
        return real_replace(source, destination, *args, **kwargs)

    with create(root, resume=True, cache_bytes=4096) as store:
        store.read((0, 0, 0), (2, 2, 2))
        with monkeypatch.context() as m:
            m.setattr(os, "fsync", fsync)
            m.setattr(os, "replace", replace)
            with pytest.raises(OSError):
                store.commit(
                    (0, 0, 0), np.full((2, 2, 2), 11), {**PROGRESS, "next_block": 4}, np.zeros((2, 2, 2), bool)
                )
        with pytest.raises(RuntimeError):
            store.read((0, 0, 0), (2, 2, 2))
        assert store.manifest == previous
        assert old_path.exists()
    with create(root, resume=True) as reopened:
        new_published = stage == "manifest_dir_fsync"
        np.testing.assert_array_equal(reopened.read((0, 0, 0), (2, 2, 2)), 11 if new_published else 7)
        np.testing.assert_array_equal(reopened.read_observed((0, 0, 0), (2, 2, 2)), not new_published)
        assert reopened.manifest["progress"] == ({**PROGRESS, "next_block": 4} if new_published else PROGRESS)
        reopened.commit((0, 0, 0), np.full((2, 2, 2), 13), {**PROGRESS, "next_block": 5})
    with create(root, resume=True) as final:
        np.testing.assert_array_equal(final.read((0, 0, 0), (2, 2, 2)), 13)
        assert final.manifest["progress"]["sweep_max_update"] == 17.5


def test_progress_only_failure_is_recoverable(tmp_path, monkeypatch):
    root = tmp_path / "field"
    seed(root)
    with create(root, resume=True) as store:
        with monkeypatch.context() as m:
            m.setattr(os, "replace", lambda *a, **kw: (_ for _ in ()).throw(OSError("rename failed")))
            with pytest.raises(OSError):
                store.save_progress({"sweep": 3})
        with pytest.raises(RuntimeError):
            store.nodes([0])
    with create(root, resume=True) as store:
        assert store.manifest["progress"] == PROGRESS
        store.save_progress({"sweep": 3, "maximum": 17.5})
    assert checkpoint(root)["progress"] == {"sweep": 3, "maximum": 17.5}


def test_atomic_json_is_strict_and_replacement_failures_preserve_old_document(tmp_path, monkeypatch):
    path = tmp_path / "STATUS.json"
    api().atomic_json(path, {"status": "checkpointed", "count": 3})
    before = path.read_bytes()
    with pytest.raises(ValueError):
        api().atomic_json(path, {"bad": np.inf})
    assert path.read_bytes() == before
    with monkeypatch.context() as m:
        m.setattr(os, "replace", lambda *a, **kw: (_ for _ in ()).throw(OSError("rename failed")))
        with pytest.raises(OSError):
            api().atomic_json(path, {"status": "new"})
    assert path.read_bytes() == before
    api().atomic_json(path, {"status": "resumed"})
    assert json.loads(path.read_bytes()) == {"status": "resumed"}


def test_parallel_read_finishes_before_commit_can_publish(tmp_path):
    root = tmp_path / "field"
    seed(root)
    entered, release, committed = threading.Event(), threading.Event(), threading.Event()
    real_sha256 = hashlib.sha256

    def blocked_hash(data=b"", **kwargs):
        if threading.current_thread().name.startswith("field-read"):
            entered.set()
            assert release.wait(5)
        return real_sha256(data, **kwargs)

    with create(root, resume=True, workers=2) as store, ThreadPoolExecutor(2) as executor:
        with patch.object(hashlib, "sha256", blocked_hash):
            reading = executor.submit(store.read, (0, 0, 0), (2, 2, 2))
            assert entered.wait(5)

            def write():
                store.commit((0, 0, 0), np.full((2, 2, 2), 9), {})
                committed.set()

            writing = executor.submit(write)
            try:
                assert not committed.wait(0.05)
                assert checkpoint(root)["progress"] == PROGRESS
            finally:
                release.set()
            np.testing.assert_array_equal(reading.result(timeout=5), 7)
            writing.result(timeout=5)
        np.testing.assert_array_equal(store.read((0, 0, 0), (2, 2, 2)), 9)


@pytest.mark.parametrize("corruption", ["symlink", "bytes"])
def test_changed_old_generation_is_rejected_before_publication(tmp_path, corruption):
    root = tmp_path / "field"
    manifest = seed(root)
    path = root / manifest["chunks"]["0-0-0"]["path"]
    with create(root, resume=True, cache_bytes=4096) as store:
        store.read((0, 0, 0), (2, 2, 2))
        if corruption == "symlink":
            target = tmp_path / "outside"
            target.write_bytes(path.read_bytes())
            path.unlink()
            path.symlink_to(target)
        else:
            path.write_bytes(b"changed")
        with pytest.raises((ValueError, OSError)):
            store.commit((0, 0, 0), np.ones((2, 2, 2)), {"next_block": 4})
        assert checkpoint(root) == manifest
        assert store.manifest == manifest
        assert path.exists()


def test_new_store_durably_publishes_its_directory_entry(tmp_path, monkeypatch):
    real_fsync = os.fsync
    parent_stat = tmp_path.stat()
    parent_synced = False

    def fsync(fd):
        nonlocal parent_synced
        current = os.fstat(fd)
        if (current.st_dev, current.st_ino) == (parent_stat.st_dev, parent_stat.st_ino):
            parent_synced = True
        return real_fsync(fd)

    monkeypatch.setattr(os, "fsync", fsync)
    with create(tmp_path / "field") as store:
        assert store.manifest["progress"] == {}
        assert parent_synced


@pytest.mark.parametrize("stage", ["payload", "manifest"])
def test_partial_writes_leave_last_checkpoint_resumable(tmp_path, monkeypatch, stage):
    root = tmp_path / "field"
    manifest = seed(root)
    real_open = Path.open

    class PartialWrite:
        def __init__(self, stream):
            self.stream = stream

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.stream.close()

        def write(self, data):
            self.stream.write(data[: len(data) // 2])
            self.stream.flush()
            raise OSError("injected partial write")

    def opened(path, mode="r", *args, **kwargs):
        stream = real_open(path, mode, *args, **kwargs)
        target = path.name.endswith(".npz.pending") if stage == "payload" else path.name.startswith(".CHECKPOINT.json.")
        return PartialWrite(stream) if mode == "xb" and target else stream

    with create(root, resume=True) as store:
        with monkeypatch.context() as m:
            m.setattr(Path, "open", opened)
            with pytest.raises(OSError, match="partial write"):
                store.commit((0, 0, 0), np.full((2, 2, 2), 99), {"next_block": 4})
        assert store.manifest == manifest
        with pytest.raises(RuntimeError):
            store.save_progress({})
    pending = {p.name: p.read_bytes() for p in root.glob("*.pending")}
    assert pending
    with create(root, resume=True) as store:
        assert store.manifest == manifest
        np.testing.assert_array_equal(store.read((0, 0, 0), (2, 2, 2)), 7)
        store.commit((0, 0, 0), np.full((2, 2, 2), 11), PROGRESS)
    assert {name: (root / name).read_bytes() for name in pending} == pending


def test_atomic_json_rejects_symlink_without_touching_target(tmp_path):
    target = tmp_path / "outside"
    target.write_bytes(b"unrelated")
    link = tmp_path / "STATUS.json"
    link.symlink_to(target)
    with pytest.raises(ValueError):
        api().atomic_json(link, {"status": "new"})
    assert target.read_bytes() == b"unrelated"


def test_retirement_failure_keeps_new_checkpoint_resumable(tmp_path, monkeypatch):
    root = tmp_path / "field"
    previous = seed(root)
    old = root / previous["chunks"]["0-0-0"]["path"]
    real_unlink = Path.unlink

    def unlink(path, *args, **kwargs):
        if path == old:
            raise OSError("injected retirement failure")
        return real_unlink(path, *args, **kwargs)

    with create(root, resume=True, cache_bytes=4096) as store:
        store.read((0, 0, 0), (2, 2, 2))
        with monkeypatch.context() as m:
            m.setattr(Path, "unlink", unlink)
            with pytest.raises(OSError, match="retirement"):
                store.commit((0, 0, 0), np.full((2, 2, 2), 9), {**PROGRESS, "next_block": 4})
        assert store.cache_nbytes == 0
        with pytest.raises(RuntimeError):
            store.read((0, 0, 0), (2, 2, 2))
    assert old.exists()
    with create(root, resume=True) as store:
        np.testing.assert_array_equal(store.read((0, 0, 0), (2, 2, 2)), 9)
        assert store.manifest["progress"] == {**PROGRESS, "next_block": 4}
    assert old.exists()


def test_generation_collision_cannot_overwrite_a_referenced_payload(tmp_path, monkeypatch):
    root = tmp_path / "field"
    manifest = seed(root)
    name = manifest["chunks"]["0-0-0"]["path"]
    original = (root / name).read_bytes()
    generation = UUID(name.split(".g")[1].split(".")[0])
    with create(root, resume=True) as store:
        monkeypatch.setattr(api().uuid, "uuid4", lambda: generation)
        with pytest.raises(FileExistsError):
            store.commit((0, 0, 0), np.full((2, 2, 2), 99), {"next_block": 4})
    assert (root / name).read_bytes() == original
    assert checkpoint(root) == manifest
    with create(root, resume=True) as store:
        np.testing.assert_array_equal(store.read((0, 0, 0), (2, 2, 2)), 7)


def test_reader_submission_is_bounded_even_when_first_batch_stalls(tmp_path, monkeypatch):
    root = tmp_path / "field"
    shape = (2, 2, 10)
    entered = threading.Barrier(3)
    release = threading.Event()
    submitted = []
    real_sha256 = hashlib.sha256

    class CountingPool(ThreadPoolExecutor):
        def submit(self, *args, **kwargs):
            submitted.append(1)
            return super().submit(*args, **kwargs)

    def blocked_hash(data=b"", **kwargs):
        if threading.current_thread().name.startswith("field-read") and not release.is_set():
            entered.wait(timeout=5)
            assert release.wait(5)
        return real_sha256(data, **kwargs)

    with create(root, shape=shape, workers=2) as store, ThreadPoolExecutor(1) as caller:
        for x in range(5):
            store.commit((0, 0, x), np.full((2, 2, 2), x + 1), {})
        monkeypatch.setattr(api(), "ThreadPoolExecutor", CountingPool)
        monkeypatch.setattr(hashlib, "sha256", blocked_hash)
        reading = caller.submit(store.read, (0, 0, 0), shape)
        try:
            entered.wait(timeout=5)
            assert len(submitted) == 2
        finally:
            release.set()
        expected = np.broadcast_to([1, 1, 2, 2, 3, 3, 4, 4, 5, 5], shape)
        np.testing.assert_array_equal(reading.result(timeout=5), expected)


def test_failed_parallel_read_drains_other_workers_before_releasing_transaction(tmp_path, monkeypatch):
    root = tmp_path / "field"
    shape = (2, 2, 4)
    second_started, first_failed, release = threading.Event(), threading.Event(), threading.Event()
    real_sha256 = hashlib.sha256
    with create(root, shape=shape, workers=2) as store, ThreadPoolExecutor(1) as caller:
        store.commit((0, 0, 0), np.ones((2, 2, 2)), {})
        store.commit((0, 0, 1), np.full((2, 2, 2), 2), {})
        first = (root / store.manifest["chunks"]["0-0-0"]["path"]).read_bytes()

        def failing_hash(data=b"", **kwargs):
            if threading.current_thread().name.startswith("field-read"):
                if data == first:
                    assert second_started.wait(5)
                    first_failed.set()
                    raise ValueError("injected reader failure")
                second_started.set()
                assert release.wait(5)
            return real_sha256(data, **kwargs)

        with monkeypatch.context() as m:
            m.setattr(hashlib, "sha256", failing_hash)
            reading = caller.submit(store.read, (0, 0, 0), shape)
            try:
                assert second_started.wait(5)
                assert first_failed.wait(5)
                with pytest.raises(TimeoutError):
                    reading.result(timeout=0.05)
            finally:
                release.set()
            with pytest.raises(ValueError, match="reader failure"):
                reading.result(timeout=5)
        store.commit((0, 0, 0), np.full((2, 2, 2), 3), {})
        np.testing.assert_array_equal(store.read((0, 0, 0), shape), np.broadcast_to([3, 3, 2, 2], shape))
