# -*- coding: utf-8 -*-
from contextlib import nullcontext

import pandas as pd
import pymupdf
import pytest
from pytest import approx

from fiadoc.parser import Page, TextBlock


@pytest.fixture(scope='module')
def page():
    file_path = 'fiadoc/tests/fixtures/page.pdf'
    doc = pymupdf.open(file_path)
    return Page(doc[0], file_path)


@pytest.mark.parametrize(
    'text, bbox, superscript, strikeout, expected, expectation',
    [
        (
            # Normal case
            'some text',
            (1, 2, 3, 4),
            None,
            None,
            TextBlock(text='some text', bbox=(1.00, 2.00, 3.00, 4.00)),
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


@pytest.mark.parametrize(
    'option, clip, expected',
    [
        (
            # Usual case
            'text',
            (110, 210, 250, 250),
            [TextBlock(text='All Officials, All Teams')]
        ),
        (
            # Has strikeout text
            'dict',
            (450, 700, 500, 710),
            [TextBlock(text='17:04.076', bbox=(460.5, 699.3, 495.3, 711.5), strikeout=True)]
        ),
        (
            # Has superscript
            'dict',
            (30, 610, 45, 623),
            [TextBlock(text='45', bbox=(37.7, 610.6, 49.9, 622.8)),
             TextBlock(text='1', bbox=(33.8, 610.7, 37.7, 618.5), superscript=True)]
        ),
        (
            # Has nothing
            'blocks',
            (200, 680, 300, 700),
            []
        ),
        (
            # Need OCR, and native `.get_text` shouldn't be able to find anything
            'blocks',
            (100, 180, 200, 200),
            []
        )
    ]
)
def test_page_native_get_text(page, option, clip, expected):
    result = page._native_get_text(option=option, clip=clip)
    assert len(result) == len(expected)
    for res_block, exp_block in zip(result, expected):
        assert res_block == exp_block


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
        assert res_block == exp_block


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


@pytest.mark.parametrize(
    'vlines, hlines, tol, allow_multiple_texts_per_cell, header_included, expected',
    [
        (
            # Simple table
            [420, 442, 455, 500],
            [652, 663, 675, 688, 699],
            2,
            None,
            True,
            pd.DataFrame(
                data=[[
                    TextBlock(text='1', bbox=(434.8, 662.8, 439.3, 675.0)),
                    TextBlock(text=''),
                    TextBlock(text='16:01:43', bbox=(464.9, 662.8, 495.3, 675.0))
                ], [
                    TextBlock(text='2', bbox=(434.8, 675.0, 439.3, 687.1)),
                    TextBlock(text=''),
                    TextBlock(text='1:16.698', bbox=(465.0, 675.0, 495.4, 687.1))
                ], [
                    TextBlock(text='3', bbox=(434.8, 687.1, 439.3, 699.3)),
                    TextBlock(text='P', bbox=(443.3, 687.1, 448.0, 699.3)),
                    TextBlock(text='1:53.862', bbox=(465.0, 687.1, 495.4, 699.3))
                ]],
                columns=['NO', '', 'TIME']
            )
        ),
        (
            # Has strikeout text in table
            [420, 442, 455, 500],
            [675, 688, 699, 711],
            2,
            None,
            False,
            pd.DataFrame(
                data=[[
                    TextBlock(text='2', bbox=(434.8, 675.0, 439.3, 687.1)),
                    TextBlock(text=''),
                    TextBlock(text='1:16.698', bbox=(465.0, 675.0, 495.4, 687.1))
                ], [
                    TextBlock(text='3', bbox=(434.8, 687.1, 439.3, 699.3)),
                    TextBlock(text='P', bbox=(443.3, 687.1, 448.0, 699.3)),
                    TextBlock(text='1:53.862', bbox=(465.0, 687.1, 495.4, 699.3))
                ], [
                    TextBlock(text='4', bbox=(434.8, 699.3, 439.3, 711.5)),
                    TextBlock(text=''),
                    TextBlock(text='17:04.076', bbox=(460.5, 699.3, 495.3, 711.5), strikeout=True)
                ]]
            )
        ),
        (
            # Allow multiple texts in the zero-th col.
            [20, 55, 150],
            [610, 623, 636, 650],
            2,
            [0],
            False,
            pd.DataFrame(
                data=[[
                    [TextBlock(text='45', bbox=(37.7, 610.6, 53.0, 622.8)),
                     TextBlock(text='1', bbox=(33.8, 610.7, 37.7, 618.5), superscript=True)],
                    TextBlock(text='Victor Martins', bbox=(63.8, 610.6, 134.1, 622.8)),
                ], [
                    [TextBlock(text='50', bbox=(37.7, 623.2, 49.9, 635.5)),
                     TextBlock(text='2', bbox=(33.8, 623.4, 37.7, 631.2), superscript=True)],
                    TextBlock(text='Ryo Hirakawa', bbox=(63.8, 623.2, 135.3, 635.5))
                ], [
                    [TextBlock(text='0', bbox=(37.7, 635.6, 43.8, 647.8))],
                    TextBlock(text='')
                ]]
            )
        )
    ]
)
def test_page_parse_table_by_grid(page, vlines, hlines, tol, allow_multiple_texts_per_cell,
                                  header_included, expected):
    result = page.parse_table_by_grid(vlines=vlines,
                                      hlines=hlines,
                                      tol=tol,
                                      allow_multiple_texts_per_cell=allow_multiple_texts_per_cell,
                                      header_included=header_included)
    pd.testing.assert_frame_equal(result,
                                  expected,
                                  check_dtype=False,
                                  check_index_type=False,
                                  check_column_type=False,
                                  check_frame_type=False,
                                  check_names=False)
