import pytest


def impl():
    import evaluation_split as m

    assert hasattr(m, "build_fibre_split"), "length-balanced fibre split missing"
    return m


def record(fid, length=20, source=None):
    return dict(fibre_id=fid, length_mm=length, source_lineage=source or [("A", str(fid))])


def test_length_strata_reserve_twenty_percent_and_are_order_independent():
    m = impl()
    records = [record(i, 0.5 if i < 50 else 20) for i in range(100)]
    a = m.build_fibre_split(records, [])
    b = m.build_fibre_split(records[::-1], [])
    assert a == b
    assert sum(r["split"] == "reserved" for r in a["fibres"]) == 20
    assert sum(r["split"] == "reserved" and r["length_mm"] >= 20 for r in a["fibres"]) == 10


def test_transitive_lineage_and_calibration_never_leak():
    m = impl()
    records = [record(1, 20, [("A", "1")]), record(2, 30, [("A", "1"), ("A", "2")]), record(3, 40, [("A", "2")])]
    a = m.build_fibre_split(records, [1])
    assert len(a["groups"]) == 1
    assert {r["split"] for r in a["fibres"]} == {"train"}
    assert a["groups"][0]["forced_calibration"]
    with pytest.raises(ValueError):
        m.build_fibre_split(records, [99])


def test_missing_lineage_is_not_claimed_independent_and_duplicate_ids_fail():
    m = impl()
    r = dict(fibre_id=1, length_mm=2, source_lineage=[])
    a = m.build_fibre_split([r], [])
    assert a["fibres"][0]["split"] == "train" and not a["fibres"][0]["lineage_verified"]
    with pytest.raises(ValueError):
        m.build_fibre_split([r, r], [])


def test_frozen_split_rejects_af_hash_and_assignment_mutations():
    m = impl()
    assert hasattr(m, "validate_fibre_split"), "split identity validation missing"
    a = m.build_fibre_split([record(i) for i in range(10)], [])
    a["afv_sha256"] = "afv"
    m.validate_fibre_split(a, "afv")
    with pytest.raises(ValueError):
        m.validate_fibre_split(a, "other")
    a["fibres"][0]["split"] = "reserved" if a["fibres"][0]["split"] == "train" else "train"
    with pytest.raises(ValueError):
        m.validate_fibre_split(a, "afv")
