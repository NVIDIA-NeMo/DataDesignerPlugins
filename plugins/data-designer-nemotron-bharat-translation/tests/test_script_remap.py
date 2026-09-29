# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from data_designer_nemotron_bharat_translation.script_remap import can_remap_script, remap_script


def test_can_remap_script_true_for_supported_pair() -> None:
    assert can_remap_script("Beng", "Mtei") is True


def test_can_remap_script_false_for_unsupported_pair() -> None:
    assert can_remap_script("Deva", "Mtei") is False
    assert can_remap_script("Beng", "Deva") is False


def test_remap_script_applies_the_bengali_to_meetei_mapping() -> None:
    assert remap_script("কা", "Beng", "Mtei") == "ꯀꯥ"


def test_remap_script_raises_for_unsupported_pair() -> None:
    with pytest.raises(ValueError, match="No script remapper"):
        remap_script("नमस्ते", "Deva", "Mtei")


def test_remap_script_preserves_fenced_code_blocks_by_default() -> None:
    text = 'কা ```label = "বাংলা"``` কা'
    assert remap_script(text, "Beng", "Mtei") == 'ꯀꯥ ```label = "বাংলা"``` ꯀꯥ'


def test_remap_script_preserves_inline_code_by_default() -> None:
    text = 'কা `label = "বাংলা"` কা'
    assert remap_script(text, "Beng", "Mtei") == 'ꯀꯥ `label = "বাংলা"` ꯀꯥ'


def test_remap_script_preserves_urls_by_default() -> None:
    text = "কা https://example.com/বাংলা কা"
    assert remap_script(text, "Beng", "Mtei") == "ꯀꯥ https://example.com/বাংলা ꯀꯥ"


def test_remap_script_can_disable_protected_span_preservation() -> None:
    text = "কা `কা` কা"
    assert remap_script(text, "Beng", "Mtei", preserve_protected_spans=False) == "ꯀꯥ `ꯀꯥ` ꯀꯥ"
