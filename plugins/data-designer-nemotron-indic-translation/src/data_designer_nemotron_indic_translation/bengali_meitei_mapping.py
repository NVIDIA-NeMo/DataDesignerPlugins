# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic Bengali-script to Meetei Mayek transliteration table (Manipuri).

A many-to-one graphemic mapping: several Bengali consonants/matras that are
graphemically distinct but phonemically merged in Meetei Mayek map to the
same target character (e.g. ``চ``/``ছ`` both map to ``ꯆ``). The mapping
never leaves the source script and never injects a third script, and digits
are converted too, so the output is script-clean by construction. Code
points are verified against Python's Unicode database (Meetei Mayek
U+ABC0..ABFF, Bengali U+0980..09FF).
"""

from __future__ import annotations

#: Core 1:1 (and many-to-one) mapping as aligned strings: BENGALI[i] -> MEITEI[i].
BENGALI = "কখগঘঙচছজঝঞটঠডঢণতথদধনপফবভমযরলশষসহৰৱৎািীুূেৈোৌৃ্ংঁঅইঈউঊ০১২৩৪৫৬৭৮৯।"
MEITEI = "ꯀꯈꯒꯘꯉꯆꯆꯖꯓꯅꯇꯊꯗꯙꯅꯇꯊꯗꯙꯅꯄꯐꯕꯚꯃꯌꯔꯂꯁꯁꯁꯍꯔꯋꯠꯥꯤꯤꯨꯨꯦꯩꯣꯧꯤ꯭ꯪꯪꯑꯏꯏꯎꯎ꯰꯱꯲꯳꯴꯵꯶꯷꯸꯹꯫"

#: Single Bengali independent vowel -> two Meetei code points.
BENGALI_MULTI = {"আ": "ꯑꯥ", "এ": "ꯑꯦ", "ঐ": "ꯑꯩ", "ও": "ꯑꯣ", "ঔ": "ꯑꯧ", "ঋ": "ꯔꯤ"}

#: Nukta letters: base + U+09BC (two code points) and precomposed forms. Applied before the table.
BENGALI_NUKTA = {
    "ড়": "ꯔ",  # RRA decomposed -> RAI
    "ড়": "ꯔ",  # RRA precomposed -> RAI
    "ঢ়": "ꯔ",  # RHA decomposed -> RAI
    "ঢ়": "ꯔ",  # RHA precomposed -> RAI
    "য়": "ꯌ",  # YYA decomposed -> YANG
    "য়": "ꯌ",  # YYA precomposed -> YANG
}

#: Dropped: visarga, standalone nukta, avagraha (no Meetei equivalent).
BENGALI_DROP = "ঃ়ঽ"

_TABLE = str.maketrans({**dict(zip(BENGALI, MEITEI, strict=True)), **BENGALI_MULTI, **{c: "" for c in BENGALI_DROP}})


def transliterate(text: str) -> str:
    """Converts Bengali-script Manipuri text into Meetei Mayek.

    Non-Bengali characters (e.g. Latin text, punctuation outside the
    mapping table) pass through unchanged.

    Args:
        text: Bengali-script source text.

    Returns:
        The Meetei Mayek transliteration of ``text``.
    """
    if not text:
        return text
    for sequence, meitei in BENGALI_NUKTA.items():
        text = text.replace(sequence, meitei)
    return text.translate(_TABLE)
