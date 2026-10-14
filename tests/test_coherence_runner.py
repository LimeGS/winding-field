import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np


def module():
    p = Path(__file__).resolve().parents[1] / "tools/run_coherence.py"
    assert p.exists(), "executable coherence ablation runner missing"
    spec = importlib.util.spec_from_file_location("coherence_runner", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


class RunnerTests(unittest.TestCase):
    def test_fixture_runs_all_five_features_and_writes_measured_cases(self):
        m = module()
        with tempfile.TemporaryDirectory() as tmp:
            r = m.synthetic_run(Path(tmp) / "new")
            self.assertTrue(all(k in r["cases"] for k in ("B1", "B2", "B3", "B3-R", "B4", "B5")))
            self.assertTrue(r["synthetic_separation_control"])
            self.assertTrue(r["synthetic_collapse_control_rejected"])
            self.assertTrue(r["cases"]["B5"]["converged"])
            self.assertTrue((Path(tmp) / "new/MANIFEST.json").exists())

    def test_cli_never_overwrites_existing_results(self):
        m = module()
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileExistsError):
                m.synthetic_run(Path(tmp))

    def test_frozen_input_binding_rejects_changed_metadata(self):
        m = module()
        self.assertTrue(hasattr(m, "validate_cached_reference"))
        import json

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            p = root / "coarse.npy"
            np.save(p, np.zeros((3, 4, 5), np.float32))
            meta = dict(
                field_sha256=m.sha256(p),
                step=1.2,
                origin_xyz=[0, 0, 0],
                spacing_xyz=[1] * 3,
                shape_zyx=[3, 4, 5],
                solver={"converged": True},
            )
            side = Path(str(p) + ".json")
            side.write_text(json.dumps(meta))
            plan = dict(baseline_sha256=m.sha256(p), baseline_metadata_sha256=m.sha256(side), calibration_step=1.2)
            m.validate_cached_reference(p, plan, (3, 4, 5))
            meta["step"] = 2
            side.write_text(json.dumps(meta))
            with self.assertRaises(ValueError):
                m.validate_cached_reference(p, plan, (3, 4, 5))


if __name__ == "__main__":
    unittest.main()
