import builtins
import hashlib
import json
import os
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import field_sources
from field_sources import CertifiedPresence, NpyObservations


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def presence_files(tmp_path, *, shape=(3, 4, 5), chunks=(2, 3, 4), coverage=None, payloads=None, compressor=None):
    root = tmp_path / "presence"
    root.mkdir()
    metadata = {
        "zarr_format": 2,
        "shape": list(shape),
        "chunks": list(chunks),
        "dtype": "|u1",
        "fill_value": 0,
        "order": "C",
        "dimension_separator": "/",
        "filters": None,
        "compressor": compressor,
    }
    (root / ".zarray").write_text(json.dumps(metadata))
    inventory = {}
    for key, raw in (payloads or {}).items():
        path = root / key
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
        inventory[key] = sha(raw)
    manifest = {
        "schema": "winding-presence-v1",
        "metadata_sha256": sha((root / ".zarray").read_bytes()),
        "shape_zyx": list(shape),
        "coverage_zyx": coverage if coverage is not None else [[[0, 0, 0], list(shape)]],
        "chunks": inventory,
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    return root, path, manifest


def open_presence(root, path, manifest=None):
    if manifest is not None:
        path.write_text(json.dumps(manifest))
    return CertifiedPresence(root, path, sha(path.read_bytes()))


@pytest.mark.parametrize("codec", ["pickle", "vlen-utf8"])
def test_presence_rejects_object_codecs_before_decoding_data(tmp_path, codec):
    root, path, _ = presence_files(tmp_path, compressor={"id": codec})
    with pytest.raises(ValueError, match="compressor|byte codec"):
        open_presence(root, path)


def test_presence_assembles_partial_edge_chunks_and_certified_zero(tmp_path):
    first = np.arange(24, dtype=np.uint8).reshape(2, 3, 4)
    edge = np.full((2, 3, 4), 173, np.uint8)
    root, path, _ = presence_files(tmp_path, payloads={"0/0/0": first.tobytes(), "1/1/1": edge.tobytes()})
    reader = open_presence(root, path)
    result, receipts = reader.read((1, 1, 1), (3, 4, 5))
    expected = np.zeros((2, 3, 4), np.uint8)
    expected[0, :2, :3] = first[1, 1:, 1:]
    expected[1, 2, 3] = 173
    np.testing.assert_array_equal(result, expected)
    assert result.dtype == np.uint8
    assert len(receipts) == 10  # Manifest, metadata, eight touched chunks.
    assert receipts[str(root / "0/0/0")] == sha(first.tobytes())
    assert all(isinstance(k, str) and len(v) == 64 and int(v, 16) >= 0 for k, v in receipts.items())
    _, again = reader.read((1, 1, 1), (3, 4, 5))
    assert again == receipts
    # Same zeros with a different certificate must not share an authority digest.
    path.write_bytes(path.read_bytes() + b"\n")
    _, other = open_presence(root, path).read((1, 1, 1), (3, 4, 5))
    assert other[str(root / "0/0/1")] != receipts[str(root / "0/0/1")]


def test_presence_does_not_fill_unknown_coverage(tmp_path):
    root, path, _ = presence_files(tmp_path, coverage=[[[0, 0, 0], [2, 3, 4]]])
    reader = open_presence(root, path)
    values, _ = reader.read((0, 0, 0), (2, 3, 4))
    assert not values.any()
    with pytest.raises(ValueError, match="coverage"):
        reader.read((1, 1, 1), (3, 4, 5))


@pytest.mark.parametrize("change", ["missing", "changed", "appeared"])
def test_presence_missing_changed_and_unmanifested_payloads_fail(tmp_path, change):
    payloads = {} if change == "appeared" else {"0/0/0": bytes(24)}
    root, path, _ = presence_files(tmp_path, payloads=payloads)
    reader = open_presence(root, path)
    chunk = root / "0/0/0"
    if change == "missing":
        chunk.unlink()
    else:
        chunk.parent.mkdir(parents=True, exist_ok=True)
        chunk.write_bytes(bytes([1]) * 24)
    with pytest.raises((ValueError, FileNotFoundError)):
        reader.read((0, 0, 0), (1, 1, 1))


def test_presence_metadata_is_rechecked_on_every_read(tmp_path):
    root, path, _ = presence_files(tmp_path)
    reader = open_presence(root, path)
    (root / ".zarray").write_bytes((root / ".zarray").read_bytes() + b" ")
    with pytest.raises(ValueError, match="metadata"):
        reader.read((0, 0, 0), (1, 1, 1))


def test_presence_hashes_and_decodes_the_same_payload_bytes(tmp_path, monkeypatch):
    root, path, _ = presence_files(tmp_path, payloads={"0/0/0": bytes([7]) * 24})
    reader = open_presence(root, path)
    read_bytes = Path.read_bytes

    def replace_after_read(p):
        raw = read_bytes(p)
        if p == root / "0/0/0":
            p.write_bytes(bytes([9]) * 24)
        return raw

    monkeypatch.setattr(Path, "read_bytes", replace_after_read)
    values, receipts = reader.read((0, 0, 0), (1, 1, 1))
    assert values.item() == 7
    assert receipts[str(root / "0/0/0")] == sha(bytes([7]) * 24)


@pytest.mark.parametrize("kind", ["manifest", "metadata"])
def test_presence_requires_pinned_hashes(tmp_path, kind):
    root, path, manifest = presence_files(tmp_path)
    if kind == "manifest":
        with pytest.raises(ValueError, match="manifest"):
            CertifiedPresence(root, path, "0" * 64)
    else:
        manifest["metadata_sha256"] = "0" * 64
        with pytest.raises(ValueError, match="metadata"):
            open_presence(root, path, manifest)


@pytest.mark.parametrize(
    "key,value",
    [
        ("schema", "unknown"),
        ("metadata_sha256", "not a hash"),
        ("shape_zyx", [3, 4, 4]),
        ("shape_zyx", [3, True, 5]),
        ("coverage_zyx", []),
        ("coverage_zyx", [[[0, 0, 0], [1, 3, 4]]]),
        ("coverage_zyx", [[[1, 0, 0], [3, 4, 5]]]),
        ("coverage_zyx", [[[0, 0, 0], [3, 4, 6]]]),
        ("coverage_zyx", [[0, 0, 0, 3, 4, 5]]),
        ("coverage_zyx", [[[0, 0, 0], [3.0, 4, 5]]]),
        ("chunks", {"../0/0": "0" * 64}),
        ("chunks", {"/0/0/0": "0" * 64}),
        ("chunks", {"00/0/0": "0" * 64}),
        ("chunks", {"0\\0\\0": "0" * 64}),
        ("chunks", {"9/0/0": "0" * 64}),
        ("chunks", {"0/0/0": "bad"}),
        ("chunks", []),
    ],
)
def test_presence_rejects_invalid_manifest_contract(tmp_path, key, value):
    root, path, manifest = presence_files(tmp_path)
    manifest[key] = value
    with pytest.raises(ValueError):
        open_presence(root, path, manifest)


def test_presence_rejects_inventory_outside_certified_boxes(tmp_path):
    root, path, manifest = presence_files(tmp_path, coverage=[[[0, 0, 0], [2, 3, 4]]])
    manifest["chunks"] = {"1/0/0": "0" * 64}
    with pytest.raises(ValueError, match="coverage"):
        open_presence(root, path, manifest)


@pytest.mark.parametrize(
    "key,value",
    [
        ("zarr_format", 3),
        ("zarr_format", 2.0),
        ("dtype", "<f4"),
        ("fill_value", 1),
        ("fill_value", False),
        ("order", "F"),
        ("dimension_separator", "."),
        ("filters", [{"id": "delta"}]),
        ("compressor", "blosc"),
        ("shape", [0, 4, 5]),
        ("shape", [3, 4]),
        ("chunks", [0, 3, 4]),
        ("chunks", [2.0, 3, 4]),
        ("chunks", [2, True, 4]),
        ("chunks", [1024, 1024, 1024]),
    ],
)
def test_presence_rejects_invalid_metadata(tmp_path, key, value):
    root, path, manifest = presence_files(tmp_path)
    metadata = json.loads((root / ".zarray").read_bytes())
    metadata[key] = value
    (root / ".zarray").write_text(json.dumps(metadata))
    manifest["metadata_sha256"] = sha((root / ".zarray").read_bytes())
    with pytest.raises(ValueError):
        open_presence(root, path, manifest)


def test_presence_requires_explicit_filter_metadata(tmp_path):
    root, path, manifest = presence_files(tmp_path)
    metadata = json.loads((root / ".zarray").read_bytes())
    del metadata["filters"]
    (root / ".zarray").write_text(json.dumps(metadata))
    manifest["metadata_sha256"] = sha((root / ".zarray").read_bytes())
    with pytest.raises(ValueError, match="metadata"):
        open_presence(root, path, manifest)


@pytest.mark.parametrize("raw", [b"[]", b"{", b'{"schema":1,"schema":2}', b'{"schema":NaN}'])
def test_presence_rejects_nonobject_or_ambiguous_json(tmp_path, raw):
    root, path, _ = presence_files(tmp_path)
    path.write_bytes(raw)
    with pytest.raises(ValueError):
        open_presence(root, path)


@pytest.mark.parametrize("relative", [".zarray", "0", "0/0/0"])
def test_presence_rejects_symlinked_metadata_or_chunk_paths(tmp_path, relative):
    root, path, _ = presence_files(tmp_path, payloads={"0/0/0": bytes(24)})
    reader = open_presence(root, path)
    source = root / relative
    moved = tmp_path / "moved"
    source.rename(moved)
    source.symlink_to(moved, target_is_directory=moved.is_dir())
    with pytest.raises(ValueError, match="path|symlink"):
        reader.read((0, 0, 0), (1, 1, 1))


def test_presence_rejects_root_redirected_after_open(tmp_path):
    root, path, _ = presence_files(tmp_path)
    reader = open_presence(root, path)
    moved = tmp_path / "moved"
    root.rename(moved)
    root.symlink_to(moved, target_is_directory=True)
    with pytest.raises(ValueError, match="path|symlink"):
        reader.read((0, 0, 0), (1, 1, 1))


def test_presence_does_not_certify_zero_through_a_nondirectory(tmp_path):
    root, path, _ = presence_files(tmp_path)
    reader = open_presence(root, path)
    (root / "0").write_bytes(b"corrupt chunk directory")
    with pytest.raises(ValueError, match="path|directory"):
        reader.read((0, 0, 0), (1, 1, 1))


def test_presence_rejects_wrong_decoded_chunk_length(tmp_path):
    root, path, _ = presence_files(tmp_path, payloads={"0/0/0": bytes(23)})
    with pytest.raises(ValueError, match="size"):
        open_presence(root, path).read((0, 0, 0), (1, 1, 1))


def test_presence_optional_codec_decodes_full_padded_edge_chunk(tmp_path):
    numcodecs = pytest.importorskip("numcodecs")
    codec = numcodecs.Zlib(level=1)
    root, path, _ = presence_files(
        tmp_path, payloads={"1/1/1": codec.encode(bytes([43]) * 24)}, compressor=codec.get_config()
    )
    volume, _ = open_presence(root, path).read((2, 3, 4), (3, 4, 5))
    assert volume.item() == 43


def test_raw_presence_does_not_require_optional_codecs(tmp_path, monkeypatch):
    original = builtins.__import__

    def without_codecs(name, *args, **kwargs):
        if name == "numcodecs":
            raise ImportError("optional codec unavailable")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_codecs)
    root, path, _ = presence_files(tmp_path, payloads={"0/0/0": bytes([4]) * 24})
    assert open_presence(root, path).read((0, 0, 0), (1, 1, 1))[0].item() == 4
    metadata = json.loads((root / ".zarray").read_bytes())
    metadata["compressor"] = {"id": "zlib", "level": 1}
    (root / ".zarray").write_text(json.dumps(metadata))
    manifest = json.loads(path.read_bytes())
    manifest["metadata_sha256"] = sha((root / ".zarray").read_bytes())
    with pytest.raises(ImportError, match="numcodecs|codec|data"):
        open_presence(root, path, manifest).read((0, 0, 0), (1, 1, 1))


def npy_files(tmp_path, **changes):
    shape = (8, 9, 10)
    arrays = {
        "target": np.arange(np.prod(shape) * 3, dtype=np.float64).reshape(*shape, 3) / 16,
        "weights": np.full(shape, 2, np.float32),
        "observed": np.indices(shape).sum(axis=0) % 2 == 0,
    }
    arrays.update(changes)
    paths, hashes = {}, {}
    for key, array in arrays.items():
        paths[key] = tmp_path / (key + ".npy")
        np.save(paths[key], array)
        hashes[key] = sha(paths[key].read_bytes())
    return paths, hashes, arrays


def loader(paths, hashes):
    return NpyObservations(paths["target"], paths["weights"], paths["observed"], hashes)


def test_npy_bounded_crops_have_owned_typed_outputs_and_disjoint_receipts(tmp_path):
    paths, hashes, arrays = npy_files(tmp_path)
    source = loader(paths, hashes)
    assert source.shape == (8, 9, 10)
    assert source.identity == hashes
    identity = source.identity
    identity["target"] = "0" * 64
    assert source.identity == hashes
    sl = np.s_[1:3, 2:5, 3:7]
    result = source((1, 2, 3), (3, 5, 7))
    for actual, key, dtype in zip(result[:3], arrays, [np.float32, np.float32, np.bool_]):
        np.testing.assert_array_equal(actual, arrays[key][sl].astype(dtype))
        assert actual.dtype == dtype and actual.flags.c_contiguous
        actual[...] = 0
        np.testing.assert_array_equal(np.load(paths[key]), arrays[key])
    again = source((1, 2, 3), (3, 5, 7))[3]
    assert result[3] == again
    other = source((0, 0, 0), (1, 1, 1))[3]
    assert any(key not in other for key in again)
    assert all(again[key] == other[key] for key in again.keys() & other.keys())
    assert all(len(value) == 64 and int(value, 16) >= 0 for value in again.values())


def test_npy_pickling_reopens_pinned_files_without_serializing_memmaps(tmp_path):
    paths, hashes, _ = npy_files(tmp_path)
    source = loader(paths, hashes)
    raw = pickle.dumps(source)
    assert len(raw) < 2048
    reopened = pickle.loads(raw)
    for a, b in zip(source((0, 0, 0), (2, 3, 4)), reopened((0, 0, 0), (2, 3, 4))):
        if isinstance(a, dict):
            assert a == b
        else:
            np.testing.assert_array_equal(a, b)
    changed = np.load(paths["weights"], mmap_mode="r+")
    changed[0, 0, 0] = 3
    changed.flush()
    with pytest.raises(ValueError, match="hash"):
        pickle.loads(raw)


def test_npy_loader_runs_in_a_fresh_child(tmp_path):
    paths, hashes, _ = npy_files(tmp_path)
    source = loader(paths, hashes)
    worker = (
        "import json,pickle,sys; "
        "source=pickle.loads(sys.stdin.buffer.read()); "
        "result=source((0,0,0),(2,3,4)); "
        "print(json.dumps({'shape':source.shape,'identity':source.identity,'receipts':result[3]}))"
    )
    result = subprocess.run(
        [sys.executable, "-c", worker],
        input=pickle.dumps(source),
        cwd=tmp_path,
        env=os.environ | {"PYTHONPATH": str(Path(field_sources.__file__).parent)},
        capture_output=True,
        timeout=15,
        check=True,
    )
    assert not result.stderr
    assert json.loads(result.stdout) == {
        "shape": list(source.shape),
        "identity": source.identity,
        "receipts": source((0, 0, 0), (2, 3, 4))[3],
    }


def test_npy_fortran_and_big_endian_sources_are_cropped_without_axis_changes(tmp_path):
    target = np.asfortranarray(np.arange(8 * 9 * 10 * 3, dtype=">f4").reshape(8, 9, 10, 3))
    paths, hashes, _ = npy_files(tmp_path, target=target)
    actual, _, _, _ = loader(paths, hashes)((2, 3, 4), (5, 6, 7))
    np.testing.assert_array_equal(actual, target[2:5, 3:6, 4:7])
    assert actual.dtype == np.float32 and actual.flags.c_contiguous


def test_npy_rejects_npz_without_leaking_its_open_archive(tmp_path, monkeypatch):
    paths, hashes, _ = npy_files(tmp_path)
    with paths["target"].open("wb") as stream:
        np.savez(stream, target=np.zeros((1, 1, 1, 3)))
    hashes["target"] = sha(paths["target"].read_bytes())
    opened = []
    original = np.load

    def capture_archive(*args, **kwargs):
        result = original(*args, **kwargs)
        opened.append(result)
        return result

    monkeypatch.setattr(np, "load", capture_archive)
    with pytest.raises(ValueError, match="NPY"):
        loader(paths, hashes)
    assert all(archive.zip is None for archive in opened)


@pytest.mark.parametrize("key", ["target", "weights", "observed"])
def test_npy_changed_crop_changes_receipt_even_with_unchanged_timestamp(tmp_path, key):
    paths, hashes, _ = npy_files(tmp_path)
    source = loader(paths, hashes)
    bounds = ((0, 0, 0), (2, 3, 4))
    before = source(*bounds)
    st = paths[key].stat()
    mapped = np.load(paths[key], mmap_mode="r+")
    mapped[0, 0, 0] = 7 if key != "observed" else False
    mapped.flush()
    os.utime(paths[key], ns=(st.st_atime_ns, st.st_mtime_ns))
    after = source(*bounds)
    assert before[3].keys() == after[3].keys()
    assert before[3] != after[3]
    assert source.identity == hashes


def test_npy_crop_receipt_detects_changes_lost_by_float32_cast(tmp_path):
    paths, hashes, _ = npy_files(tmp_path)
    source = loader(paths, hashes)
    before = source((1, 1, 1), (2, 2, 2))
    mapped = np.load(paths["target"], mmap_mode="r+")
    mapped[1, 1, 1, 0] = np.nextafter(mapped[1, 1, 1, 0], np.inf)
    mapped.flush()
    after = source((1, 1, 1), (2, 2, 2))
    np.testing.assert_array_equal(before[0], after[0])
    assert before[3] != after[3]


@pytest.mark.parametrize("change", ["replace", "truncate", "header"])
def test_npy_detects_replaced_truncated_or_reheadered_source(tmp_path, change):
    paths, hashes, arrays = npy_files(tmp_path)
    source = loader(paths, hashes)
    path = paths["weights"]
    if change == "replace":
        replacement = tmp_path / "replacement.npy"
        np.save(replacement, arrays["weights"] + 1)
        replacement.replace(path)
    elif change == "truncate":
        with path.open("r+b") as stream:
            stream.truncate(128)
    else:
        raw = path.read_bytes().replace(b"(8, 9, 10)", b"(9, 8, 10)")
        path.write_bytes(raw)
    with pytest.raises(ValueError, match="changed"):
        source((0, 0, 0), (1, 1, 1))


@pytest.mark.parametrize(
    "key,value",
    [
        ("target", np.zeros((8, 9, 10))),
        ("target", np.zeros((8, 9, 10, 2))),
        ("target", np.zeros((8, 9, 10, 3), np.int32)),
        ("target", np.zeros((8, 9, 10, 3), np.complex64)),
        ("target", np.zeros((0, 9, 10, 3))),
        ("weights", np.ones((8, 9, 10), np.int32)),
        ("weights", np.ones((8, 9, 9))),
        ("observed", np.ones((8, 9, 10), np.uint8)),
        ("observed", np.ones((8, 9, 9), bool)),
    ],
)
def test_npy_validates_source_shapes_and_dtypes(tmp_path, key, value):
    paths, hashes, _ = npy_files(tmp_path, **{key: value})
    with pytest.raises(ValueError, match="shape|dtype"):
        loader(paths, hashes)


@pytest.mark.parametrize(
    "key,value", [("target", np.nan), ("target", np.inf), ("weights", 0), ("weights", -1), ("weights", np.inf)]
)
def test_npy_validates_values_only_in_requested_crop(tmp_path, key, value):
    paths, hashes, _ = npy_files(tmp_path)
    mapped = np.load(paths[key], mmap_mode="r+")
    mapped[7, 8, 9] = value
    mapped.flush()
    hashes[key] = sha(paths[key].read_bytes())
    source = loader(paths, hashes)
    source((0, 0, 0), (1, 1, 1))
    with pytest.raises(ValueError, match="finite|positive"):
        source((7, 8, 9), (8, 9, 10))


@pytest.mark.parametrize("value", [1e300, 1e-300])
def test_npy_rejects_weights_not_representable_as_positive_float32(tmp_path, value):
    paths, hashes, _ = npy_files(tmp_path, weights=np.full((8, 9, 10), value))
    with pytest.raises(ValueError, match="finite|positive"):
        loader(paths, hashes)((0, 0, 0), (1, 1, 1))


@pytest.mark.parametrize("change", ["missing", "extra", "bad", "mismatch"])
def test_npy_requires_exact_pinned_source_hashes(tmp_path, change):
    paths, hashes, _ = npy_files(tmp_path)
    if change == "missing":
        del hashes["target"]
    elif change == "extra":
        hashes["unknown"] = "0" * 64
    else:
        hashes["target"] = "bad" if change == "bad" else "0" * 64
    with pytest.raises(ValueError, match="hash"):
        loader(paths, hashes)


@pytest.mark.parametrize(
    "lo,hi",
    [
        ((-1, 0, 0), (1, 1, 1)),
        ((0, 0, 0), (9, 10, 11)),
        ((1, 0, 0), (1, 1, 1)),
        ((0, 0), (1, 1, 1)),
        ((0.0, 0, 0), (1, 1, 1)),
        ((False, 0, 0), (1, 1, 1)),
    ],
)
def test_both_readers_reject_invalid_half_open_bounds(tmp_path, lo, hi):
    root, path, _ = presence_files(tmp_path)
    paths, hashes, _ = npy_files(tmp_path)
    for read in (open_presence(root, path).read, loader(paths, hashes)):
        with pytest.raises(ValueError, match="bound|integer"):
            read(lo, hi)


def fixed_loader(path, digest, shape=(2, 3, 4)):
    assert hasattr(field_sources, "NpyFixedValues"), "pinned fixed-value NPY loader is missing"
    return field_sources.NpyFixedValues(path, digest, shape)


@pytest.mark.parametrize("dtype", ["float32", "float64", ">f4", ">f8"])
@pytest.mark.parametrize("order", ["C", "F"])
def test_npy_fixed_crops_preserve_precision_nan_zero_and_owned_storage(tmp_path, dtype, order):
    precision = 1.0 + 2 ** (-40 if np.dtype(dtype).itemsize == 8 else -20)
    large = 1e100 if np.dtype(dtype).itemsize == 8 else 1e30
    expected = np.array([[[np.nan, 0.0], [-0.0, precision]], [[large, -2.5], [0.0, np.nan]]], dtype=dtype)
    values = np.full((2, 3, 4), np.nan, dtype=dtype, order=order)
    values[:, 1:3, 1:3] = expected
    path = tmp_path / "fixed.npy"
    np.save(path, values)
    digest = sha(path.read_bytes())
    source = fixed_loader(str(path) if order == "F" else path, digest, [2, 3, 4])
    assert source.shape == (2, 3, 4)
    assert source.identity == {"path": str(path.absolute()), "sha256": digest}
    detached = source.identity
    detached.update(path="elsewhere", sha256="0" * 64)
    assert source.identity == {"path": str(path.absolute()), "sha256": digest}
    crop = source((0, 1, 1), (2, 3, 3))
    np.testing.assert_array_equal(crop, expected)
    np.testing.assert_array_equal(np.signbit(crop), np.signbit(expected))
    assert crop.dtype == expected.dtype
    assert crop.flags.owndata and crop.flags.c_contiguous and crop.flags.writeable
    crop[:] = 9
    np.testing.assert_array_equal(source((0, 1, 1), (2, 3, 3)), expected)
    np.testing.assert_array_equal(np.load(path, allow_pickle=False), values)


@pytest.mark.parametrize("value", [np.inf, -np.inf])
def test_npy_fixed_rejects_infinity_only_in_requested_crop(tmp_path, value):
    path = tmp_path / "fixed.npy"
    values = np.zeros((2, 3, 4), np.float64)
    values[1, 2, 3] = value
    np.save(path, values)
    source = fixed_loader(path, sha(path.read_bytes()))
    np.testing.assert_array_equal(source((0, 0, 0), (1, 1, 2)), [[[0.0, 0.0]]])
    with pytest.raises(ValueError, match="inf|finite"):
        source((1, 2, 3), (2, 3, 4))


@pytest.mark.parametrize("dtype", [np.float16, np.int32, np.uint8, bool, np.complex64, object, "U1", [("x", "f4")]])
def test_npy_fixed_rejects_unsupported_source_dtypes(tmp_path, dtype):
    path = tmp_path / "fixed.npy"
    np.save(path, np.zeros((2, 3, 4), dtype=dtype))
    with pytest.raises(ValueError):
        fixed_loader(path, sha(path.read_bytes()))


@pytest.mark.parametrize("shape", [(2, 3, 3), (2, 3), (0, 3, 4), (2.0, 3, 4), (True, 3, 4), (2, 3, 4, 1)])
def test_npy_fixed_requires_matching_positive_integer_zyx_shape(tmp_path, shape):
    path = tmp_path / "fixed.npy"
    np.save(path, np.zeros((2, 3, 4)))
    with pytest.raises(ValueError, match="shape|integer"):
        fixed_loader(path, sha(path.read_bytes()), shape)


@pytest.mark.parametrize("digest", ["bad", "A" * 64, "0" * 64, None])
def test_npy_fixed_requires_valid_pinned_full_file_sha256(tmp_path, digest):
    path = tmp_path / "fixed.npy"
    np.save(path, np.zeros((2, 3, 4)))
    with pytest.raises(ValueError, match="SHA256|hash"):
        fixed_loader(path, digest)


def test_npy_fixed_rejects_npz_and_closes_archive(tmp_path, monkeypatch):
    path = tmp_path / "fixed.npy"
    with path.open("wb") as stream:
        np.savez(stream, fixed=np.zeros((2, 3, 4)))
    opened = []
    original = np.load

    def capture_archive(*args, **kwargs):
        archive = original(*args, **kwargs)
        opened.append(archive)
        return archive

    monkeypatch.setattr(np, "load", capture_archive)
    with pytest.raises(ValueError, match="NPY"):
        fixed_loader(path, sha(path.read_bytes()))
    assert opened and all(archive.zip is None for archive in opened)


@pytest.mark.parametrize("change", ["replace", "truncate", "header"])
def test_npy_fixed_detects_replaced_truncated_or_reheadered_source(tmp_path, change):
    path = tmp_path / "fixed.npy"
    np.save(path, np.zeros((2, 3, 4)))
    source = fixed_loader(path, sha(path.read_bytes()))
    if change == "replace":
        replacement = tmp_path / "replacement.npy"
        replacement.write_bytes(path.read_bytes())
        replacement.replace(path)
    elif change == "truncate":
        with path.open("r+b") as stream:
            stream.truncate(128)
    else:
        stat = path.stat()
        path.write_bytes(path.read_bytes().replace(b"(2, 3, 4)", b"(3, 2, 4)"))
        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    with pytest.raises(ValueError, match="changed"):
        source((0, 0, 0), (1, 1, 1))


@pytest.mark.parametrize(
    "lo,hi",
    [
        ((-1, 0, 0), (1, 1, 1)),
        ((0, 0, 0), (3, 3, 4)),
        ((1, 0, 0), (1, 1, 1)),
        ((0, 0), (1, 1, 1)),
        ((0.0, 0, 0), (1, 1, 1)),
        ((False, 0, 0), (1, 1, 1)),
        ((0, 0, 0), (True, 1, 1)),
    ],
)
def test_npy_fixed_rejects_invalid_half_open_bounds(tmp_path, lo, hi):
    path = tmp_path / "fixed.npy"
    np.save(path, np.zeros((2, 3, 4)))
    source = fixed_loader(path, sha(path.read_bytes()))
    with pytest.raises(ValueError, match="bound|integer"):
        source(lo, hi)


def test_npy_fixed_pickle_reopens_pinned_source_without_array_payload(tmp_path):
    path = tmp_path / "fixed.npy"
    values = np.full((16, 16, 16), np.nan)
    values[1, 2, 3] = 1.0 + 2**-40
    np.save(path, values)
    source = fixed_loader(path, sha(path.read_bytes()), values.shape)
    raw = pickle.dumps(source)
    assert len(raw) < 1024
    reopened = pickle.loads(raw)
    assert reopened.shape == values.shape and reopened.identity == source.identity
    np.testing.assert_array_equal(reopened((1, 2, 3), (2, 3, 4)), [[[1.0 + 2**-40]]])
    mapped = np.load(path, mmap_mode="r+")
    mapped[1, 2, 3] = 2
    mapped.flush()
    with pytest.raises(ValueError, match="hash"):
        pickle.loads(raw)
