# -*- coding: utf-8 -*-
from contextlib import nullcontext

import pytest

from fiadoc.parser import TextBlock

@pytest.mark.parametrize(
    'text, bbox, superscript, strikeout, expected, expectation',
    [
        (
            # Normal case
            'some text',
            (1, 2, 3, 4),
            False,
            False,
            TextBlock(text='some text', bbox=(1.00, 2.00, 3.00, 4.00)),
            nullcontext()
        ),
        (
            # `text` is None, so should raise TypeError
            None,
            None,
            None,
            None,
            None,
            pytest.raises(TypeError)
        ),
        (
            # `text` is not string, so should raise TypeError
            -5,
            None,
            None,
            None,
            None,
            pytest.raises(TypeError)
        ),
        (
            # Negative bbox values, so should raise ValueError
            'something',
            (1, 1, 2, -10),
            False,
            False,
            None,
            pytest.raises(ValueError)
        ),
        (
            # Strikeout text
            'text',
            None,
            False,
            True,
            TextBlock(text='text', strikeout=True),
            nullcontext()
        )
    ]
)
def test_textblock(text, bbox, superscript, strikeout, expected, expectation):
    with expectation:
        block = TextBlock(
            text=text,
            bbox=bbox,
            superscript=superscript,
            strikeout=strikeout
        )
        assert block == expected


def test_textblock_attribute_mutation():
    tb = TextBlock(text='initial', bbox=(1, 2, 3, 4))
    with pytest.raises(TypeError):
        tb.text = 1
    with pytest.raises(TypeError):
        tb.bbox = 'invalid'
    with pytest.raises(ValueError):
        tb.bbox = (1, 2, 3, -4)

    tb.text = 'changed'
    assert tb.text == 'changed'
    tb.bbox = (5, 6, 7, 8)
    assert tb.bbox == (5, 6, 7, 8)
    tb.superscript = True
    assert tb.superscript is True
    tb.strikeout = True
    assert tb.strikeout is True
