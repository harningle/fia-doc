# -*- coding: utf-8 -*-
from contextlib import nullcontext

import numpy as np
import pandas as pd
import pymupdf
import pytest
from pytest import approx

from fiadoc.parser import Page, TextBlock
from fiadoc.parser.parser import BaseParser
from fiadoc.utils import download_pdf


@pytest.fixture(scope='module')
def page():
    file_path = 'fiadoc/tests/fixtures/page.pdf'
    doc = pymupdf.open(file_path)
    return Page(doc[0], file_path)


@pytest.mark.parametrize(
    'clip, col_min_gap, min_black_line_length, expected',
    [
        (
            # Normal case
            (0, 280, 595, 295),
            1.1,
            0.9,
            [TextBlock(text='No.', bbox=(33.7, 281.1, 51.8, 291.0)),
             TextBlock(text='Driver', bbox=(63.6, 281.4, 96.7, 291.2)),
             TextBlock(text='Nat', bbox=(168.3, 281.1, 187.0, 291.0)),
             TextBlock(text='Team', bbox=(203.1, 281.1, 232.8, 291.0)),
             TextBlock(text='Constructor', bbox=(408.9, 280.9, 473.5, 291.0))]
        ),
        (
            # Need OCR
            (420, 650, 500, 665),
            1.1,
            0.9,
            [TextBlock(text='NO', bbox=(430.5, 653.2, 442.4, 660.8)),
             TextBlock(text='TIME', bbox=(471.7, 653.2, 491.2, 660.6))]
        )
    ]
)
def test_detect_cols(page, clip, col_min_gap, min_black_line_length, expected):
    result = BaseParser._detect_cols(
        page,
        clip=clip,
        col_min_gap=col_min_gap,
        min_black_line_length=min_black_line_length
    )
    assert result == expected


@pytest.mark.parametrize('url, expected', [
    (
        # One penalty
        'https://www.fia.com/system/files/decision-document/2026_miami_grand_prix_-_final_sprint_qualifying_classification.pdf',
        ["Car 23 - Lap time of 1.30.988 deleted - Track limits - Stewards' document no. 28"]
    ),
    (
        # Multiple penalties
        'https://www.fia.com/system/files/documents/doc_100_-_2026_monaco_grand_prix_-_revised_final_race_classification.pdf',
        ["Car 43 - 5 second time penalty - Speeding in the pit lane - Stewards' decision no. 72",
         "Car 18 - 5 second time penalty - Track limits - Stewards' decision no. 77",
         "Car 27 - 10 second time penalty - Causing a collision - Stewards' decision no. 89",
         "Car 11 - 10 second time penalty - False start - Stewards' decision no. 82"]
    ),
    (
        # No PENALTIES table
        'https://www.fia.com/system/files/decision-document/2026_australian_grand_prix_-_final_qualifying_classification.pdf',
        []
    ),
])
def test_parse_penalties(url, expected, tmp_path):
    download_pdf(url, tmp_path / 'classification.pdf')
    assert BaseParser._parse_penalties(tmp_path / 'classification.pdf') == expected
