# -*- coding: utf-8 -*-
from contextlib import nullcontext

import pymupdf
import pytest
from pytest import approx

from fiadoc.parser import Page, TextBlock


@pytest.mark.parametrize(
    'text, bbox, superscript, strikeout, expected_repr, expectation',
    [
        (
            # Normal case
            'some text',
            (1, 2, 3, 4),
            None,
            None,
            'TextBlock(text="some text", bbox=(1.00, 2.00, 3.00, 4.00))',
            nullcontext()
        ),
        (
            # `text` is None, so should raise ValueError
            None,
            None,
            None,
            None,
            None,
            pytest.raises(ValueError)
        ),
        (
            # Negative bbox values, so should raise ValueError
            'something',
            (1, 1, 2, -10),
            None,
            None,
            None,
            pytest.raises(ValueError)
        ),
        (
            # Strikeout text
            'text',
            None,
            False,
            True,
            'TextBlock(text="text", strikeout=True)',
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


@pytest.fixture(scope='module')
def page():
    file_path = 'fiadoc/tests/fixtures/page.pdf'
    doc = pymupdf.open(file_path)
    return Page(doc[0], file_path)


@pytest.mark.parametrize(
    'option, clip, small_area, expected',
    [
        (
            # Usual case
            'text',
            (110, 210, 250, 250),
            False,
            [TextBlock(text='All Officials, All Teams')]
        ),
        (
            # Usual case
            'dict',
            (110, 210, 250, 250),
            False,
            [TextBlock(text='All Officials, All Teams', bbox=(114.5, 208.3, 227.0, 220.5))]
        ),
        (
            # Has superscript
            'dict',
            (30, 610, 45, 623),
            False,
            [TextBlock(text='45', bbox=(37.7, 610.6, 49.9, 622.8)),
             TextBlock(text='1', bbox=(33.8, 610.7, 37.7, 618.5), superscript=True)]
        ),
        (
            # Has nothing
            'blocks',
            (200, 680, 300, 700),
            False,
            []
        ),
        (
            # Need OCR
            'blocks',
            (100, 180, 200, 200),
            False,
            [TextBlock(text='The Stewards', bbox=(114.2, 183.4, 182.5, 194.6))]
        ),
        (
            # Has strikeout text
            'dict',
            (450, 700, 500, 710),
            False,
            [TextBlock(text='17:04.076', bbox=(460.5, 699.3, 495.3, 711.5), strikeout=True)]
        )
    ]
)
def test_page_get_text(page, option, clip, small_area, expected):
    result = page.get_text(option=option, clip=clip, small_area=small_area)
    assert len(result) == len(expected)
    for res_block, exp_block in zip(result, expected):
        assert res_block.text == exp_block.text
        if exp_block.bbox:
            assert is_bbox_almost_equal(
                (res_block.x0, res_block.y0, res_block.x1, res_block.y1),
                (exp_block.x0, exp_block.y0, exp_block.x1, exp_block.y1)
            )
        assert res_block.superscript == exp_block.superscript
        assert res_block.strikeout == exp_block.strikeout


@pytest.mark.parametrize(
    'clip, max_thickness, min_length, rgb, expected',
    [
        (
            # Two valid lines and one very short line which should be ignored
            (50, 150, 200, 260),
            2,
            0.8,
            50,
            [173, 254]
        ),
        (
            # No black line
            (300, 290, 400, 300),
            2,
            0.8,
            50,
            []
        ),
        (
            # A mix of black line and black rectangle should return only line
            (0, 250, 400, 300),
            2,
            0.8,
            50,
            [254]
        ),
        (
            # A real example from lap times PDF
            (450, 650, 510, 750),
            2,
            0.5,
            192,
            [663, 706]
        )
    ]
)
def test_page_search_for_black_lines(page, clip, max_thickness, min_length, rgb, expected):
    black_lines = page.search_for_black_lines(clip=clip,
                                              max_thickness=max_thickness,
                                              min_length=min_length,
                                              rgb=rgb)
    assert len(black_lines) == len(expected)
    for line_y_coord, exp_line_y_coord in zip(black_lines, expected):
        assert line_y_coord == approx(exp_line_y_coord, abs=1)


def is_bbox_almost_equal(
        bbox1: tuple[float, float, float, float],
        bbox2: tuple[float, float, float, float],
        atol: float = 1
) -> bool:
    return all(a == approx(b, abs=atol) for a, b in zip(bbox1, bbox2))
