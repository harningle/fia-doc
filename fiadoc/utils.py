import importlib.resources
import os
import re
import tempfile
import uuid
from functools import cached_property
from pathlib import Path
from string import printable
from typing import Literal, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymupdf
import pytesseract
import requests
from PIL import Image

from .models.foreign_key import SessionEntryForeignKeys
from .models.lap import LapImport, LapObject

os.environ['TESSDATA_PREFIX'] = importlib.resources.files('fiadoc').as_posix()
DPI = 300  # Works the best for tesseract OCR

rc = {'figure.figsize': (4, 3),
      'axes.facecolor': 'white',  # Remove background colour
      'axes.grid': False,         # Turn on grid
      'axes.linewidth': '0.2',
      'axes.edgecolor': '0',      # Set axes edge color to be black
      'font.size': 2,
      'xtick.major.size': 1,
      'xtick.major.width': 0.2,
      'ytick.major.size': 1,
      'ytick.major.width': 0.2}
plt.rcdefaults()
plt.rcParams.update(rc)


def duration_to_millisecond(s: str | None) -> dict[str, str | int] | None:
    """Convert a time duration string to milliseconds

    >>> duration_to_millisecond('1:36:48.076')
    {'_type': 'timedelta', 'milliseconds': 5808076}
    >>> duration_to_millisecond('17:39.564')
    {'_type': 'timedelta', 'milliseconds': 1059564}
    >>> duration_to_millisecond('12.345')
    {'_type': 'timedelta', 'milliseconds': 12345}
    """
    if s is None:
        return None

    match s.count(':'):
        case 0:  # 12.345
            assert re.match(r'\d+\.\d+', s), f'{s} is not a valid time duration'
            sec, millisec = s.split('.')
            return {
                '_type': 'timedelta',
                'milliseconds': int(sec) * 1000 + int(millisec)
            }

        case 1:  # 1:23.456
            if m := re.match(r'(?P<minute>\d+):(?P<sec>\d+)\.(?P<millisec>\d+)', s):
                minute = int(m.group('minute'))
                sec = int(m.group('sec'))
                millisec = int(m.group('millisec'))
                return {
                    '_type': 'timedelta',
                    'milliseconds': minute * 60000 + sec * 1000 + millisec
                }
            else:
                raise ValueError(f'{s} is not a valid time duration')

        case 2:  # 1:23:45.678
            if m := re.match(r'(?P<hour>\d+):(?P<minute>\d+):(?P<sec>\d+)\.(?P<millisec>\d+)', s):
                hour = int(m.group('hour'))
                minute = int(m.group('minute'))
                sec = int(m.group('sec'))
                millisec = int(m.group('millisec'))
                return {
                    '_type': 'timedelta',
                    'milliseconds': hour * 3600000 + minute * 60000 + sec * 1000 + millisec
                }
            else:
                raise ValueError(f'{s} is not a valid time duration')

        case _:
            raise ValueError(f'{s} is not a valid time duration')


def time_to_timedelta(d: str) -> pd.Timedelta:
    """Parse a date string or a time duration string to pd.Timedelta

    TODO: not clear to me. May be confusing later. Either need better documentation, or make them
          into separate functions.

    There can be two possible input formats:

    1. hh:mm:ss, e.g. 18:05:42. This is simply the local calendar time
    2. mm:ss.SSS, e.g. 1:24.160. This is the lap time
    """
    n_colon = d.count(':')
    if n_colon == 2:  # noqa: PLR2004
        h, m, s = d.split(':')
        return pd.Timedelta(hours=int(h), minutes=int(m), seconds=int(s))
    elif n_colon == 1:
        m, s = d.split(':')
        s, ms = s.split('.')
        return pd.Timedelta(minutes=int(m), seconds=int(s), milliseconds=int(ms))
    else:
        raise ValueError(f'unknown date format: {d}')


def download_pdf(url: str, out_path: str | os.PathLike) -> None:
    """
    Download a PDF file from the given URL. This downloads PDFs for testing. This is a temporary
    solution. Will be deleted when Philipp's package is ready
    """
    resp = requests.get(url)
    with open(out_path, 'wb') as f:
        f.write(resp.content)
    return


def quali_lap_times_to_json(df, year, round_no, session) -> list[dict]:
    lap_data = []
    # TODO: first lap's lap time is calendar time, not lap time, so drop it
    # Lap No. can be missing (e.g.#47)
    df = df[(df.lap_no >= 2) | df.lap_no.isna()].copy()  # noqa: PLR2004
    df.lap_time = df.lap_time.apply(duration_to_millisecond)
    for q in [1, 2, 3]:
        temp = df[df.Q == q].copy()
        temp['lap'] = temp.apply(
            lambda x: LapObject(
                number=x.lap_no,
                time=x.lap_time,
                is_deleted=x.lap_time_deleted,
                is_entry_fastest_lap=x.is_fastest_lap
            ),
            axis=1
        )
        temp = temp.groupby('car_no')[['lap']].agg(list).reset_index()
        temp['session_entry'] = temp['car_no'].map(
            lambda x: SessionEntryForeignKeys(
                year=year,
                round=round_no,
                session=f'Q{q}' if session == 'quali' else f'SQ{q}',
                car_number=x
            )
        )
        temp['lap_data'] = temp.apply(
            lambda x: LapImport(
                object_type="Lap",
                foreign_keys=x['session_entry'],
                objects=x['lap']
            ).model_dump(exclude_unset=True),
            axis=1
        )
        lap_data.extend(temp['lap_data'].tolist())
    return lap_data


class Page:
    def __init__(self, page: pymupdf.Page):
        self._pymupdf_page = page
        self.drawings = page.get_drawings()
        self.crossed_out_text = self.get_strikeout_text()
        self.tempdir = Path(tempfile.mkdtemp('fiadoc'))

    def __getattr__(self, name: str):
        return getattr(self._pymupdf_page, name)

    @cached_property
    def ocr_page(self) -> pymupdf.Page:
        """OCR the entire page (as if it's an image) and save the result as a new PDF file

        :return: OCR-ed page as a native `pymupdf.Page` (, so `.get_text` is the native `.get_text`
                 w/o OCR. Otherwise, may be stuck in an infinite loop?)
        """
        random_filename = uuid.uuid4().hex  # TODO: to avoid name collision when multithreading (?)
        self.get_pixmap(dpi=DPI).pil_save(self.tempdir / f'{random_filename}.png')
        with open(self.tempdir / f'{random_filename}.pdf', 'wb') as f:
            f.write(pytesseract.image_to_pdf_or_hocr(
                str(self.tempdir / f'{random_filename}.png'),
                config=f'--dpi {DPI} -c tessedit_create_pdf=1')
            )  # See https://stackoverflow.com/a/73580016/12867291 for "-c tessedit_create_pdf=1"
        return pymupdf.open(self.tempdir / f'{random_filename}.pdf')[0]  # TODO: memory safe?

    def show_page(self) -> None:
        """May not working well. For debug process only

        See https://github.com/pymupdf/PyMuPDF-Utilities/blob/master/table-analysis/show_image.py
        """
        pix = self.get_pixmap(dpi=DPI)
        img = np.ndarray([pix.h, pix.w, 3], dtype=np.uint8, buffer=pix.samples_mv)
        plt.figure(dpi=DPI)
        plt.imshow(img, extent=(0, pix.w * 72 / DPI, pix.h * 72 / DPI, 0))
        plt.show()
        return

    def search_for_header(self, keyword: str) -> Optional[pymupdf.Rect]:
        """Return the location of header (like "Qualifying Session Final Classification")

        We try to locate the header by:

        1. simple search for the keyword, such as "Final Classification"
        2. if not found, OCR the page, and search again for the same keyword
        3. if not found again, search for a grey-ish image that is very wide and has reasonable
           height to be a header image
        4. if still not found, return `None`

        :param keyword: Page header text, e.g. "Provisional Classification"
        :return: The bounding box of the header as `pymupdf.Rect`
        """
        # Simple keyword search
        found = []
        for text in self.get_text('blocks'):
            if keyword in text[4]:
                found.append(text)
        if (n := len(found)) >= 2:  # noqa: PLR2004
            raise ParsingError(f'Expect only one "{keyword}" on p. {self.number} in '
                               f'{self.parent.name}. Found {n}: {found}')
        elif len(found) == 1:
            return pymupdf.Rect(found[0][:4])

        # If not found above, search the OCR-ed page
        for text in self.ocr_page.get_text('blocks'):
            if keyword in text[4]:
                found.append(text)
        if (n := len(found)) >= 2:  # noqa: PLR2004
            raise ParsingError(f'Expect only one "{keyword}" on p. {self.number} in '
                               f'{self.parent.name}. Found {n} on the OCR-ed page: {found}')
        elif len(found) == 1:
            return pymupdf.Rect(found[0][:4])

        # If still not found, search for a header image (see #26)
        """
        Basically we go through all image objects on the page, and keep the ones that:

        1. are very wide (80+% of the page width), and
        2. have reasonable height (~2% of the page height), and
        2. have grey-ish background colour (10% around #B8B8B8)

        In principle, only the header image can meet these criteria.
        """
        images = []
        for img in self.drawings:
            if ((img['rect'].width > self.bound()[2] * 0.8)
                    and np.isclose(img['rect'].height, self.bound()[3] * 0.02, rtol=0.1)
                    and np.isclose(img['fill'], [0.72, 0.72, 0.72], rtol=0.1).all()):
                images.append(img)
        assert len(images) <= 1, f'found more than one header image on page {self.number} in ' \
                                 f'{self.parent.name}'
        if images:
            return images[0]['rect']
        return None

    def search_for(self, text: str, **kwargs) -> str | list | dict:
        """`pymupdf.Page.search_for`, with OCR"""
        # Usual search
        if results := self._pymupdf_page.search_for(text, **kwargs):
            return results

        # If nothing found, OCR the page and search again
        if self.ocr_page is None:
            self.save_ocr()
        return self.ocr_page.search_for(text, **kwargs)

    @staticmethod
    def _clean_get_text_results(
            results: str | list | dict,
            option: Literal['text', 'words', 'blocks', 'dict'],
            expected: Optional[re.Pattern]
    ) -> str | list | dict:
        r"""Remove stuff that do not match the `expected` regex from `.get_text` results

        :param results: Return value from `.get_text`
        :param option: The `option` parameter in `.get_text`. Only support "text", "words",
                       "blocks", or "dict"
        :param expected: Will remove results that do not match with this regex. E.g., when we
                         extract texts in the lap time col., `expected` would probably be
                         `re.compile(r'\d+:\d+\.\d+')`.
        :return: The cleaned results, or `None` if no results match the `expected` regex
        """
        match option:
            case 'text':
                temp = ''.join([c for c in results if c in printable]).strip()
                if re.match(expected, temp) if expected else True:
                    return temp
                else:
                    return ''
            case 'words' | 'blocks':
                temp = []
                for res in results:
                    text = ''.join([c for c in res[4] if c in printable]).strip()
                    if re.match(expected, text) if expected else True:
                        temp.append((*res[:4], text, *res[5:]))
                if temp:
                    return temp
                else:
                    return []
            case 'dict':
                cleaned_blocks = []
                for block in results['blocks']:
                    cleaned_lines = []
                    for line in block['lines']:
                        cleaned_spans = []
                        for span in line['spans']:
                            span['text'] = ''.join([c for c in span['text'] if c in printable])
                            if re.match(expected, span['text']) if expected else True:
                                cleaned_spans.append(span)
                        if cleaned_spans:
                            line['spans'] = cleaned_spans
                            cleaned_lines.append(line)
                    if cleaned_lines:
                        block['lines'] = cleaned_lines
                        cleaned_blocks.append(block)
                if cleaned_blocks:
                    results['blocks'] = cleaned_blocks
                    return results
                else:
                    return {'blocks': []}
            case _:
                raise ValueError(f'`option` must be one of "text", "words", "blocks", or "dict". '
                                 f'No support for "{option}"')

    def get_text(
            self,
            option: str = 'text',
            clip: Optional[tuple[float, float, float, float]] = None,
            lang: Optional[Literal['f1', 'eng']] = 'f1',
            expected: Optional[re.Pattern] = None,
            **kwargs
    ) -> str | list | dict:
        r"""This is `pymupdf.Page.get_text` w/ OCR functionality

        This method does the native `.get_text` first. If no text that regex matches `expected` is
        found, we proceed with OCR.

        When `clip` is specified, we only OCR the clipped area instead of the whole page. And we
        assume the clipped area is small and contains only a single line of text. That is, we use
        `--psm 7` in tessaract. See
        https://github.com/tesseract-ocr/tesseract/blob/main/doc/tesseract.1.asc#options.

        When `clip` is not provided, we OCR the whole page. Note that after OCR, the text can be
        slightly off from the original text. E.g. a letter that is originally located in coord.
        (10, 20) can be in (11, 19) after OCR. For this reason, we probably never want to call this
        method without specifying `clip`

        :param option: When `clip` is specified, `option` can only be "text", "words", "blocks", or
                       "dict". Otherwise, it can be any valid `option` in `pymupdf.Page.get_text`.
        :param clip: (x0, y0, x1, y1)
        :param lang: Which tesseract language to use for OCR. Default is our own fine-tuned model
                     "f1" which is better for small clipped area. The other option is tesseract's
                     default English "eng".
        :param expected: If provided, only texts that match this regex will be returned. Default is
                         return everything w/o filtering. E.g., when we extract texts in the lap
                         time col., we want `expected` to be `re.compile(r'\d+:\d+\.\d+')`.
        :param kwargs: Other keyword arguments to pass to `pymupdf.Page.get_text`
        :return: Same type as the return of `pymupdf.Page.get_text`
        """
        def _return_empty() -> str | list | dict:
            """Helper function to return empty results"""
            match option:
                case 'text':
                    return ''
                case 'words' | 'blocks':
                    return []
                case 'dict':
                    return {'blocks': []}
                case _:
                    raise ValueError(
                        f'Native `.get_text` does not find anything. For OCR, `option` must be '
                        f'one of "text", "words", "blocks", or "dict". Got "{option}"'
                    )

        def _result_is_empty(res: str | list | dict) -> bool:
            """Check if `.get_text` results are empty"""
            if isinstance(res, str):
                return not res.strip()
            elif isinstance(res, list):
                return not any([r[4].strip() for r in res])
            elif isinstance(res, dict):
                return not any([span['text'].strip()
                                for block in res['blocks']
                                for line in block['lines']
                                for span in line['spans']])
            else:
                raise ValueError(f'Unknown type of results: {type(res)}. Expected str, list, or '
                                 f'dict. Got {res}')

        # Try simple search first
        results = self._pymupdf_page.get_text(option, clip=clip, **kwargs)
        results = self._clean_get_text_results(results, option, expected)
        if not _result_is_empty(results):
            return results

        # If `clip` is not provided, we OCR the whole page
        if clip is None:
            results = self.ocr_page.get_text(option, **kwargs)
            results = self._clean_get_text_results(results, option, expected)
            if _result_is_empty(results):
                _return_empty()
            else:
                return results

        # If we have `clip`, only OCR the clipped area. First check if any black pixels
        """
        We first check if there is any black pixel (RGB < 50) in the clipped area. If not, return
        empty immediately. This is because all texts are black(ish). We check this for two reasons:

        1. it's much faster to skip OCR if we already know there is no text in the clipped area
        2. tessaract quality is really bad. If you give it a all grey pure colour image, it can
           spit out some random text like "-". This breaks most of our parsing
        """
        pixmap = self.get_pixmap(clip=clip, dpi=DPI)
        pixmap_arr = np.ndarray(
            [pixmap.height, pixmap.width, 3], dtype=np.uint8, buffer=pixmap.samples
        ).copy()  # `samples_mv` memory error on Mac/Windows?
        if not np.any(np.all(pixmap_arr <= 50, axis=2)):  # noqa: PLR2004
            return _return_empty()

        # Replace light grey pixel (RGB > 200) with white. This improves OCR quality a lot
        pixmap_arr[np.all(pixmap_arr >= 200, axis=2)] = 255  # noqa: PLR2004
        random_filename = uuid.uuid4().hex
        Image.fromarray(pixmap_arr).save(self.tempdir / f'{random_filename}.png')

        # OCR the clipped area
        if lang not in {'f1', 'eng'}:
            raise ValueError(f'`lang` can only be "f1" or "eng". Got "{lang}"')
        text = pytesseract.image_to_string(
            str(self.tempdir / f'{random_filename}.png'),
            config=f'--psm 7 --dpi {DPI} -l {lang}'
        )

        # Some simply cleaning
        """
        During fine tuning, we use ".", "-", "_", ",", "|", and "—" as placeholder for empty string
        (, as tesseract does not support empty string as ground truth). Sometimes table boundaries
        will also be mistaken as these chars. So we remove them when they are the leading or
        trailing chars. When they appear in the middle, they probably are part of the text, so keep
        them. We also remove all non-printable chars.
        """
        text = ''.join([c for c in text.strip() if c in printable]).strip()
        text = re.sub(r'^[.\-_,|—]+$', '', text)
        text = re.sub(r'[.\-_,|—]+$', '', text).strip()
        text = re.sub(r'^[.\-_,|—]+', '', text).strip()

        if not text:
            _return_empty()
        if expected and (not re.match(expected, text)):
            _return_empty()

        # If code reaches here, we got some non-empty text (either matches `expected` or there is
        # no `expected` provided). Return the text in appropriate type, depending on `option`. See
        # https://pymupdf.readthedocs.io/en/latest/textpage.html
        match option:
            case 'text':
                return text
            case 'words':
                return [(*clip, text, -1, -1, -1)]
            case 'blocks':
                return [(*clip, text, -1, -1)]
            case 'dict':
                return {
                    'blocks': [{
                        'bbox': clip,
                        'lines': [{
                            'bbox': clip,
                            'spans': [{
                                'bbox': clip,
                                'text': text
                            }]
                        }]
                    }]
                }
            case _:
                raise ValueError(f'Native `.get_text` does not find anything. For OCR, `option`'
                                 f'must be one of "text", "words", "blocks", or "dict". Got '
                                 f'"{option}"')

    def get_drawings_in_bbox(self, bbox: tuple[float, float, float, float], tol: float = 1) \
            -> list:
        """Get all drawings in the given bounding box

        :param bbox: (x0, y0, x1, y1)
        :param tol: tolerance in pixels. If a drawing is outside the bbox by only `tol` pixels,
                    it will be included. Default is one pixel
        :return: A list of drawings
        """
        drawings = []
        for drawing in self.drawings:
            rect = drawing['rect']
            if rect.y0 >= bbox[1] - tol and rect.y1 <= bbox[3] + tol \
                    and rect.x0 >= bbox[0] - tol and rect.x1 <= bbox[2] + tol:
                drawings.append(drawing)
        return drawings

    def get_strikeout_text(self) -> list[tuple[pymupdf.Rect, str]]:
        """Get all strikeout texts and their locations in the page

        See https://stackoverflow.com/a/74582342/12867291.

        :return: A list of tuples, where each tuple is the bbox and text of a strikeout text
        """
        # Get all strikeout lines
        lines = []
        paths = self.drawings  # Strikeout lines are in fact vector graphics. To be more precise,
        for path in paths:     # they are short rectangles with very small height
            for item in path['items']:
                if item[0] == 're':  # If a graphic is a rect., check its height: absolute height
                    rect = item[1]   # should < 1px, and have some sizable width relative to height
                    if (rect.width > 2 * rect.height) and (rect.height < 1):
                        lines.append(rect)

        # Get all texts on this page
        # TODO: the O(n^2) here can probably be optimised later
        words = self.get_text('words')
        strikeout = []
        for rect in lines:
            for w in words:  # `w` is a iterable `(x0, y0, x1, y1, text)`
                text_rect = pymupdf.Rect(w[:4])     # Location/bbox of the word
                if text_rect.intersects(rect):      # If the word's location intersects with a
                    strikeout.append((rect, w[4]))  # strikeout line, it's a strikeout text
        return strikeout

    def parse_table_by_grid(
            self,
            vlines: list[tuple[float, float]],
            hlines: list[tuple[float, float]]
    ) -> tuple[pd.DataFrame, list[tuple[int, int, str]], list[tuple[int, int, bool]]]:
        """Manually parse the table cell by cell, defined by lines separating the columns and rows

        PyMuPDF does have a built-in table parser, but it's bad for many reasons:

        1. it does not always respect the `clip` parameter: sometimes it gets table header/rows
           outside the clip area
        2. we can't force it to ignore the table header. E.g., for some short table (e.g. 2 rows),
           it may get the first row as the header, and the second row as the data. However, we
           sometimes want to treat all rows as data, not header. There is no way to force this
        3. sometimes it will get an empty column, if the horizontal gap between two columns are too
           big. This is true even if we manually specify `vertical_lines`

        For these reasons, whenever we know the exact positions of rows and columns, we parse the
        table manually use this function. This function returns three things:

        1. the usual parsed df.
        2. a list of tuples to identify is a cell (i, j) is crossed out, in the format of
           (i, j, True or False)
        3. a list of tuples for the superscript for a cell (i, j), in the format of
           (i, j, superscript text)

        :param vlines: List of left and right x-coords. of the cols
        :param hlines: List of top and bottom y-coords. of the rows
        :return: A usual df., a list of superscript cells, a list of crossed out cells
        """
        cells = []
        superscripts = []
        crossed_out = []
        for i, row_sep in enumerate(hlines):
            row = []
            for j, col_sep in enumerate(vlines):
                t = row_sep[0]
                b = row_sep[1]
                l = col_sep[0]
                r = col_sep[1]

                # Get text in the cell. Since we need to check the superscript, we need `'dict'`
                # See https://pymupdf.readthedocs.io/en/latest/recipes-text.html
                cell = self.get_text('dict', clip=(l, t, r, b))
                spans = []
                for block in cell['blocks']:
                    for line in block['lines']:
                        for span in line['spans']:
                            if span['text'].strip():
                                bbox = span['bbox']
                                # Need to check if the found text is indeed in the cell's bbox.
                                # PyMuPDF is notoriously bad for not respecting `clip` parameter.
                                # We give two pixels tolerance
                                if bbox[0] >= l - 2 and bbox[2] <= r + 2 \
                                        and bbox[1] >= t - 2 and bbox[3] <= b + 2:
                                    spans.append(span)

                # If we don't find any text in the cell, it's empty. This is OK, e.g. the quali.
                # lap time table's pit column is always empty for non-pit laps
                if not spans:
                    row.append('')
                    continue

                # Check if any superscript
                # See https://pymupdf.readthedocs.io/en/latest/recipes-text.html#how-to-analyze-
                # font-characteristics for font flags
                match len(spans):
                    case 1:
                        row.append(spans[0]['text'].strip())
                    case 2:
                        for span in spans:
                            match span['flags']:
                                case 0:
                                    row.append(span['text'].strip())
                                case 1:
                                    superscripts.append((i, j, span['text'].strip()))
                                case _:
                                    raise ValueError(
                                        f'Unknown font flags for cell at ({l, t, r, b})'
                                    )
                    case _:
                        raise ValueError(f'Unknown span for cell at ({l, t, r, b})')

                # Check if the cell is crossed out
                for rect, text in self.crossed_out_text:
                    if rect.intersects((l, t, r, b)):
                        assert text == row[-1], \
                            f'Found a crossed out text at the location of ({l, t, r, b}) with ' \
                            f'text "{row[-1]}", but the crossed out text is "{text}" and '\
                            f"doesn't match with the cell's text"
                        crossed_out.append((i, j, True))
            cells.append(row)

        df = pd.DataFrame(cells, columns=None, index=None)
        return df, superscripts, crossed_out


class ParsingError(Exception):
    pass
