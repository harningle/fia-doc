# -*- coding: utf-8 -*-
from contextlib import nullcontext

import pytest

from fiadoc.utils import TextBlock


@pytest.mark.parametrize(
    'text, bbox, superscript, strikeout, expected_repr, expectation',
    [
        (
            'some text',
            (1, 2, 3, 4),
            None,
            None,
            'TextBlock(text="some text", bbox=(1.00, 2.00, 3.00, 4.00))',
            nullcontext()
        ),
        (
            None,
            None,
            None,
            None,
            None,
            pytest.raises(ValueError)
        ),
        (
            'text',
            None,
            'some superscript',
            ['strikeout', 'text'],
            "TextBlock(text=\"text\", superscript=['some superscript'], "
            "strikeout=['strikeout', 'text'])",
            nullcontext()
        ),
    ]
)
def test_textblock(text, bbox, superscript, strikeout, expected_repr, expectation):
    with expectation:
        block = TextBlock(
            text=text,
            bbox=bbox,
            superscript=superscript,
            strikeout=strikeout
        )
        assert repr(block) == expected_repr
        if bbox:
            assert block.l == block.x0 == bbox[0]
            assert block.t == block.y0 == bbox[1]
            assert block.r == block.x1 == bbox[2]
            assert block.b == block.y1 == bbox[3]
