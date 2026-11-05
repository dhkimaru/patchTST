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
