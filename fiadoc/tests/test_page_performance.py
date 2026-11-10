"""Regression coverage for page-local native extraction and lazy OCR loading."""
import subprocess
import sys

import pymupdf
import pytest

from fiadoc.parser import Page


@pytest.mark.parametrize('rotation', [0, 90, 180, 270])
@pytest.mark.parametrize('option', ['text', 'words', 'blocks', 'dict'])
@pytest.mark.parametrize('clip', [
    None,
    (110, 210, 250, 250),
    (30, 610, 45, 623),  # Partial glyphs and a superscript.
    (450, 700, 500, 710),  # Strikeout text.
    (100, 180, 200, 200),  # Image text must still trigger OCR.
    (-20, -20, 10, 10),
    (10, 10, 10, 10),
])
def test_reused_extraction_matches_pymupdf(rotation, option, clip):
    file = 'fiadoc/tests/fixtures/page.pdf'
    with pymupdf.open(file) as document:
        native = document[0]
        page = Page(native, file)
        # Populate the cache before changing the rotation and clip.
        page._get_native_text(option, clip=(30, 610, 45, 623))
        native.set_rotation(rotation)
        for sort in [False, True]:
            expected = native.get_text(option, clip=clip, sort=sort)
            assert page._get_native_text(option, clip=clip, sort=sort) == expected
            assert native.rotation == rotation


@pytest.mark.parametrize('flags', [0, pymupdf.TEXTFLAGS_DICT, pymupdf.TEXT_INHIBIT_SPACES])
def test_reused_extraction_preserves_flags_and_superscripts(flags):
    with pymupdf.open('fiadoc/tests/fixtures/page.pdf') as document:
        native = document[0]
        page = Page(native, 'page.pdf')
        for clip in [(30, 610, 45, 623), (33, 610, 39, 619), (450, 700, 500, 710)]:
            expected = native.get_text('dict', clip=clip, flags=flags)
            assert page._get_native_text('dict', clip=clip, flags=flags) == expected


def test_supplied_textpage_keeps_pymupdf_clipping_semantics():
    with pymupdf.open('fiadoc/tests/fixtures/page.pdf') as document:
        native = document[0]
        page = Page(native, 'page.pdf')
        textpage = native.get_textpage()
        clip = (110, 210, 250, 250)
        # PyMuPDF deliberately ignores clip when the caller supplies a TextPage.
        expected = native.get_text(clip=clip, textpage=textpage)
        assert expected != native.get_text(clip=clip)
        assert page._get_native_text('text', clip=clip, textpage=textpage) == expected


def test_native_parsing_does_not_import_ocr_and_factory_is_a_singleton():
    code = '''
import sys
from types import ModuleType
from unittest.mock import Mock
import pymupdf
from fiadoc.parser import Page
from fiadoc.parser.page import get_ocr_instance

with pymupdf.open('fiadoc/tests/fixtures/page.pdf') as document:
    page = Page(document[0], 'page.pdf')
    assert page.get_text(clip=(110, 210, 250, 250))[0].text == 'All Officials, All Teams'
assert not {'paddle', 'paddleocr', 'paddlex'}.intersection(sys.modules)
module = ModuleType('paddleocr')
module.PaddleOCR = Mock()
sys.modules['paddleocr'] = module
assert get_ocr_instance() is get_ocr_instance()
module.PaddleOCR.assert_called_once_with(
    lang='en', use_doc_orientation_classify=False,
    use_doc_unwarping=False, use_textline_orientation=False)
'''
    subprocess.run([sys.executable, '-c', code], check=True)
