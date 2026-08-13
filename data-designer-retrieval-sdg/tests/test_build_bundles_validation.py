from pathlib import Path

import pytest

from data_designer_retrieval_sdg.chunking import build_bundles


def test_build_bundles_rejects_zero_bundle_size():
    with pytest.raises(ValueError, match="bundle_size must be positive"):
        build_bundles([Path("a.txt"), Path("b.txt")], bundle_size=0)


def test_build_bundles_rejects_negative_bundle_size():
    with pytest.raises(ValueError, match="bundle_size must be positive"):
        build_bundles([Path("a.txt"), Path("b.txt")], bundle_size=-1)
