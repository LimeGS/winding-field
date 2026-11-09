import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

import field_store as fs


@pytest.mark.parametrize("budget", [0, 10000])
def test_fortran_chunk_sparse_gather_does_not_flatten_whole_chunk(tmp_path, monkeypatch, budget):
    class NoRavel(np.ndarray):
        def ravel(self, *args, **kwargs):
            pytest.fail("sparse gather flattened the whole Fortran chunk")

    decode = fs._decode

    def checked_decode(raw, shape):
        delta, offset, mask = decode(raw, shape)
        assert delta.flags.f_contiguous and not delta.flags.c_contiguous
        return delta.view(NoRavel), offset, mask

    monkeypatch.setattr(fs, "_decode", checked_decode)
    values = np.asfortranarray(np.arange(60, dtype=np.float64).reshape(3, 4, 5))
    with fs.BlockStore(tmp_path / "field", values.shape, 5, {}, cache_bytes=budget) as store:
        store.commit((0, 0, 0), values, {})
        plan = store.prepare_nodes([[7, 59, 7, 0]])
        np.testing.assert_array_equal(store.read_nodes_plan(plan), [[7, 59, 7, 0]])
        np.testing.assert_array_equal(store.nodes([23]), [23])


@pytest.mark.parametrize("budget", [0, 1, 600, 10000])
def test_committed_decode_reused_only_within_budget(tmp_path, budget):
    with fs.BlockStore(tmp_path / "field", (2, 2, 2), 2, {}, cache_bytes=budget) as store:
        for number in (2, 3):
            with patch.object(fs, "_decode", wraps=fs._decode) as decode:
                store.commit((0, 0, 0), np.full((2, 2, 2), number), {})
                assert decode.call_count == 1
                before = store.stats["loads"]
                np.testing.assert_array_equal(store.nodes([0, 7, 7]), number)
                assert decode.call_count == (1 if budget >= 600 else 2)
                assert store.stats["loads"] - before == (0 if budget >= 600 else 1)
                assert store.cache_nbytes <= budget
                assert set(store._cache) <= {store.manifest["chunks"]["0-0-0"]["path"]}
        store.commit((0, 0, 0), np.zeros((2, 2, 2)), {})
        assert store.cache_nbytes == 0


@pytest.mark.parametrize("stage", ["before", "after", "cleanup", "budget"])
def test_failed_publication_never_retains_either_generation(tmp_path, monkeypatch, stage):
    root = tmp_path / "field"
    with fs.BlockStore(root, (1, 1, 1), 1, {}, cache_bytes=10000) as store:
        store.commit((0, 0, 0), np.array([[[2]]]), {})
        old = root / store.manifest["chunks"]["0-0-0"]["path"]
        publish, unlink = fs._atomic_json_bytes, Path.unlink

        def fail_publish(path, raw):
            if stage == "after":
                publish(path, raw)
            raise OSError("injected publication")

        def fail_unlink(path, *args, **kwargs):
            if path == old:
                raise OSError("injected cleanup")
            return unlink(path, *args, **kwargs)

        with monkeypatch.context() as m:
            if stage in ("before", "after"):
                m.setattr(fs, "_atomic_json_bytes", fail_publish)
            elif stage == "cleanup":
                m.setattr(Path, "unlink", fail_unlink)
            else:
                store.max_bytes = 0
            with pytest.raises((OSError, fs.StorageBudgetError)):
                store.commit((0, 0, 0), np.array([[[9]]]), {})
        assert store.cache_nbytes == 0 and not store._cache
        with pytest.raises(RuntimeError):
            store.nodes([0])
    with fs.BlockStore(root, (1, 1, 1), 1, {}, resume=True) as store:
        np.testing.assert_array_equal(store.nodes([0]), 9 if stage in ("after", "cleanup") else 2)


def test_manifest_serialized_once_and_same_bytes_admitted_published(tmp_path):
    with fs.BlockStore(tmp_path / "field", (1, 1, 1), 1, {}) as store:
        with (
            patch.object(fs, "_json_bytes", wraps=fs._json_bytes) as encode,
            patch.object(store, "_admit", wraps=store._admit) as admit,
            patch.object(fs, "_atomic_json_bytes", wraps=fs._atomic_json_bytes) as publish,
        ):
            store.commit((0, 0, 0), np.array([[[1]]]), {"cursor": 2})
        manifests = [call for call in encode.call_args_list if "chunks" in call.args[0]]
        assert len(manifests) == 1
        raw = publish.call_args.args[1]
        assert json.loads(raw) == store.manifest
        assert (store.root / "CHECKPOINT.json").read_bytes() == raw
        assert admit.call_args.args[0] == len(raw) + store.manifest["chunks"]["0-0-0"]["bytes"]


def test_node_plan_edges_duplicates_snapshot_empty_and_live_values(tmp_path):
    with fs.BlockStore(tmp_path / "field", (3, 3, 5), 2, {}, workers=2) as store:
        nodes = np.array([[44, 0, 44], [2, 10, 0]])
        plan = store.prepare_nodes(nodes)
        nodes[:] = 1
        assert plan.nbytes >= sum(a.nbytes for a in (plan.starts, plan.order, plan.local))
        for a in (plan.starts, plan.order, plan.local):
            with pytest.raises(ValueError):
                a.setflags(write=True)
        for value in (3, 6):
            store.commit((1, 1, 2), np.array([[[value]]]), {})
            np.testing.assert_array_equal(store.read_nodes_plan(plan), [[value, 0, value], [0, 0, 0]])
        assert store.read_nodes_plan(store.prepare_nodes(np.empty((0, 2), int))).shape == (0, 2)
        with pytest.raises(ValueError, match="geometry"):
            store.read_nodes_plan(None)
    with fs.BlockStore(tmp_path / "other", (3, 3, 5), 1, {}) as other:
        with pytest.raises(ValueError, match="geometry"):
            other.read_nodes_plan(plan)
