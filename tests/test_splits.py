from __future__ import annotations

import pytest

from ald_lab.data.splits import SplitError, load_split


def test_split_v2_is_disjoint_and_seals_test():
    split = load_split()
    assert (len(split.train), len(split.validation), len(split.test)) == (51, 11, 11)
    assert split.partition_of("standard_trace_071") == "excluded"
    assert split.partition_of("standard_trace_001") == "train"
    assert split.partition_of("standard_trace_080") == "test"
    assert len(split.runs("train", "validation")) == 62
    with pytest.raises(SplitError):
        split.runs("test")
    assert len(split.runs("test", include_test=True)) == 11


def test_overlapping_split_is_rejected(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("split_id: bad\ntrain: [a, b]\nvalidation: [b]\ntest: []\n", encoding="utf-8")
    with pytest.raises(SplitError):
        load_split(path)


def test_side_study_splits_keep_validation_and_test():
    """Learning-curve subsets nest inside split_v2's training runs; recent-window runs come just before validation."""
    from pathlib import Path

    import yaml

    from ald_lab.data.splits import DEFAULT_SPLIT

    base = load_split()
    folder = Path(DEFAULT_SPLIT).parent
    previous: set[str] = set()
    for k in (10, 20, 31, 42):
        split = load_split(folder / f"split_v2_lc{k}.yaml")
        assert len(split.train) == k and set(split.train) <= set(base.train) and previous <= set(split.train)
        assert split.validation == base.validation and split.test == base.test
        previous = set(split.train)
    for fold in (2, 3, 4):
        split = load_split(folder / f"split_v2_recent{fold}.yaml")
        assert len(split.train) == 20 and split.test == base.test
        assert max(split.train) < min(split.validation)
        assert not set(split.train) & set(split.validation)
        earlier = sorted(r for r in base.train if r < min(split.validation))
        assert sorted(split.train) == earlier[-20:]
    experiments = Path(DEFAULT_SPLIT).parents[1] / "experiments"
    for name in [f"lc{k}" for k in (10, 20, 31, 42, 51)] + [f"recent{f}" for f in (2, 3, 4)] + ["res05", "cycleonly"]:
        doc = yaml.safe_load((experiments / f"g1_virtual_sensor_{name}.yaml").read_text(encoding="utf-8"))
        assert doc["study"] and doc["task"] == "virtual_sensor"
        split_name = doc["data"]["split"]
        assert (folder / f"{split_name}.yaml").is_file()


def test_recent_window_lengths_and_second_draw():
    from pathlib import Path

    from ald_lab.data.splits import DEFAULT_SPLIT

    base = load_split()
    folder = Path(DEFAULT_SPLIT).parent
    for fold, k in ((4, 10), (4, 30), (4, 40), (3, 30)):
        split = load_split(folder / f"split_v2_recent{fold}_{k}.yaml")
        earlier = sorted(r for r in base.train if r < min(split.validation))
        assert sorted(split.train) == earlier[-k:] and split.test == base.test
    previous: set[str] = set()
    for k in (20, 31, 42):
        split = load_split(folder / f"split_v2_lc{k}b.yaml")
        assert len(split.train) == k and previous <= set(split.train) and split.validation == base.validation
        assert set(split.train) != set(load_split(folder / f"split_v2_lc{k}.yaml").train)
        previous = set(split.train)
