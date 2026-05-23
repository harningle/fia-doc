"""Mocked exception-path tests for the parser layer

Covers representative raise sites without requiring real PDF fixtures:

    * `ValueError` when `Page.get_text` is called with no native text and no clip
    * `ValueError` for invalid session passed to a parser constructor
    * `ParsingError` when parse_table_by_grid finds multiple texts in one cell
    * `OCRError` when the OCR predict() call returns an unexpected shape
"""
from unittest.mock import MagicMock

import pymupdf
import pytest

from fiadoc.parser import Page, PracticeParser
from fiadoc.parser.page import OCRError, ParsingError


@pytest.fixture(scope='module')
def page():
    file_path = 'fiadoc/tests/fixtures/page.pdf'
    doc = pymupdf.open(file_path)
    return Page(doc[0], file_path)


def test_value_error_when_no_native_text_and_no_clip(page, monkeypatch):
    monkeypatch.setattr(page, '_native_get_text', lambda *a, **k: [])
    with pytest.raises(ValueError, match='`clip` is.*not provided'):
        page.get_text(option='blocks', clip=None)


def test_value_error_on_invalid_session(tmp_path):
    fake_pdf = tmp_path / 'classification.pdf'
    fake_pdf.write_bytes(b'%PDF-1.4\n')
    with pytest.raises(ValueError, match='Invalid session'):
        PracticeParser(fake_pdf, None, 2025, 1, 'invalid_session')


def test_parsing_error_on_multiple_texts_in_one_cell(page):
    # The fixture page has a superscript "1" alongside "45" inside (30, 610, 45, 623). W/o
    # `allow_multiple_texts_per_cell`, both the main text and the superscript (two in total) fall
    # into one cell -> `ParsingError`
    with pytest.raises(ParsingError, match='multiple texts'):
        page.parse_table_by_grid(vlines=[20, 55],
                                 hlines=[610, 623],
                                 tol=2,
                                 header_included=False)


def test_ocr_error_on_unexpected_predict_shape(page, monkeypatch):
    monkeypatch.setattr(page, '_native_get_text', lambda *a, **k: [])
    fake_ocr = MagicMock()
    fake_ocr.predict.return_value = ['unexpected', 'shape']
    monkeypatch.setattr('fiadoc.parser.page.get_ocr_instance', lambda: fake_ocr)
    with pytest.raises(OCRError, match='Unexpected OCR results'):
        page.get_text(option='blocks', clip=(100, 180, 200, 200))
