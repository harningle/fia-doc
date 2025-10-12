import os
import re
import tempfile

# import uuid
from functools import cached_property
from pathlib import Path
from string import printable
from types import SimpleNamespace
from typing import Literal, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymupdf
import requests
from paddleocr import PaddleOCR

# from PIL import Image
from .models.foreign_key import SessionEntryForeignKeys
from .models.lap import LapImport, LapObject

DPI = 600  # Works the best for OCR
OCR = PaddleOCR(lang='en',
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False)
# Correct common OCR mistakes
# TODO: very fragile...
OCR_ERRORS = {
    'KMIH': 'KM/H',
    'L': '1',
    'I': '1'
}

rc = {'figure.figsize': (8, 6),
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
    def __init__(self, page: pymupdf.Page, file: str | os.PathLike):
        self._pymupdf_page = page
        self.file = file
        self.w = page.bound()[2]
        self.h = page.bound()[3]
        self.tempdir = Path(tempfile.mkdtemp('fiadoc'))

    def __getattr__(self, name: str):
        return getattr(self._pymupdf_page, name)

    @cached_property
    def ocred_page(self) -> SimpleNamespace:
        """OCR the entire page and return the texts with their bounding boxes

        :return: An object w/ `.get_text` and `.search_for` methods, as if it is a `pymupdf.Page`
        """
        # OCR
        pixmap = self._pymupdf_page.get_pixmap(dpi=DPI)
        # random_filename = uuid.uuid4().hex
        # pixmap.pil_save(self.tempdir / f'{random_filename}.png')
        pixmap_arr = np.ndarray(
            [pixmap.height, pixmap.width, 3], dtype=np.uint8, buffer=pixmap.samples
        )
        ocred_page = OCR.predict(pixmap_arr)[0]
        ocred_page = zip(ocred_page['rec_boxes'], ocred_page['rec_texts'])

        # Because of the DPI, the bounding boxes of texts after OCR may not be in the same
        # coordinate system as the original PDF page. Convert them back
        ocred_page = [TextBlock(
            text=OCR_ERRORS.get(i[1], i[1]),
            bbox=self._transform_bbox(
                original_bbox=self._pymupdf_page.bound(),
                new_bbox=(0, 0, pixmap.width, pixmap.height),
                bbox=tuple(i[0])
            )
        ) for i in ocred_page]
        # TODO: the confidence score should somehow be used in the future

        # Define some methods like `search_for` and `get_text`
        def search_for(
                text: str,
                clip: Optional[tuple[float, float, float, float]] = None,
                tol: float = 2
        ) -> list[pymupdf.Rect]:
            """Search for text in OCR results and return the bounding boxes as `pymupdf.Rect`

            The search is case-insensitive, same as PyMuPDF.
            """
            results = [i.bbox for i in ocred_page if text.lower() in i.text.lower()]
            if clip is None:
                clip = self._pymupdf_page.bound()
            results = [pymupdf.Rect(i) for i in results
                       if i[0] > clip[0] - tol and i[1] > clip[1] - tol
                       and i[2] < clip[2] + tol and i[3] < clip[3] + tol]
            return results

        '''
        def get_text(
                option: str,
                clip: Optional[tuple[float, float, float, float]] = None,
                tol: float = 2
        ) -> str | list | dict:
            """`pymupdf.Page.get_text` equivalent for the OCR results

            TODO: this won't work for "partial get". That is, if "Lewis Hamilton" occupies the bbox
                  (10, 10, 50, 50), and we only want the text in (20, 10, 50, 50), we should get
                  "Hamilton". But the current implementation will return nothing, as the current
                  bbox is for the entire text and we don't know the coords. of part of the text.

            :param option: The `option` parameter in `pymupdf.Page.get_text`. Only support "text",
                           "words", "blocks", or "dict"
            :param clip: (x0, y0, x1, y1). If provided, only return text in this area
            :param tol: Tolerance in pixels. Only if a text is inside the clip area with `tol` px
                        margin, it will be included in the results. Default is 2 px
            :return: The text in the specified format, depending on `option`
            """
            if clip is None:
                clip = (-1, -1, self._pymupdf_page.bound()[2] + 1,
                        self._pymupdf_page.bound()[3] + 1)

            results = []
            for result in ocred_page:
                bbox = tuple(result[:4])
                if bbox[0] > clip[0] - tol and bbox[1] > clip[1] - tol \
                        and bbox[2] < clip[2] + tol and bbox[3] < clip[3] + tol:
                    if option == 'text':
                        results.append(result[4])
                    elif option == 'words':
                        results.append((bbox[0], bbox[1], bbox[2], bbox[3], result[4]))
                    elif option == 'blocks':
                        results.append((bbox[0], bbox[1], bbox[2], bbox[3], result[4], -1, -1))
                    elif option == 'dict':
                        results.append({
                            'blocks': [{
                                'bbox': clip,
                                'lines': [{
                                    'bbox': clip,
                                    'spans': [{
                                        'bbox': clip,
                                        'text': result[4]
                                    }]
                                }]
                            }]
                        })
                    else:
                        raise ValueError(f'`option` must be one of "text", "words", "blocks", or '
                                         f'"dict" if using OCR. Got "{option}"')
            if not results:
                return '' if option == 'text' else []
            return results
        '''

        return SimpleNamespace(
            search_for=search_for,
            # get_text=get_text,
            to_list=lambda: ocred_page
        )

    def show_page(self) -> None:
        """May not work well. For debug process only

        See https://github.com/pymupdf/PyMuPDF-Utilities/blob/master/table-analysis/show_image.py
        """
        pix = self.get_pixmap(dpi=DPI)
        img = np.ndarray([pix.h, pix.w, 3], dtype=np.uint8, buffer=pix.samples_mv)
        plt.figure(dpi=DPI)
        plt.imshow(img, extent=(0, pix.w * 72 / DPI, pix.h * 72 / DPI, 0))
        plt.show()
        return

    def search_for(self, text: str, **kwargs) -> str | list | dict:
        """`pymupdf.Page.search_for`, with OCR"""
        # Usual search
        if results := self._pymupdf_page.search_for(text, **kwargs):
            return results

        # If nothing found, OCR the page and search again
        return self.ocred_page.search_for(text, **kwargs)

    @staticmethod
    def _transform_bbox(
            original_bbox: tuple[float, float, float, float],
            new_bbox: tuple[float, float, float, float],
            bbox: tuple[float, float, float, float]
    ) -> tuple[float, float, float, float]:
        """Transform the bbox from new coord. system to the original system

        :param original_bbox: The bounding box of the original PDF page
        :param new_bbox: The bounding box of the image for OCR
        :param bbox: The bounding box of the text in the new coord. system
        """
        scale_x = (original_bbox[2] - original_bbox[0]) / (new_bbox[2] - new_bbox[0])
        scale_y = (original_bbox[3] - original_bbox[1]) / (new_bbox[3] - new_bbox[1])
        return (original_bbox[0] + (bbox[0] - new_bbox[0]) * scale_x,
                original_bbox[1] + (bbox[1] - new_bbox[1]) * scale_y,
                original_bbox[0] + (bbox[2] - new_bbox[0]) * scale_x,
                original_bbox[1] + (bbox[3] - new_bbox[1]) * scale_y)

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
            small_area: bool = False,
            expected: Optional[re.Pattern] = None,
            **kwargs
    ) -> str | list | dict:
        r"""This is `pymupdf.Page.get_text` w/ OCR functionality

        This method does the native `.get_text` first. If no text that regex matches `expected` is
        found (, suppose `expected` is provided), we proceed with OCR.

        * `clip` is not None: we only OCR the clipped area instead of the whole page
        * `clip` is None: we OCR the whole page, using hOCR XML

        :param option: When `clip` is specified, `option` can only be "text", "words", "blocks", or
                       "dict". Otherwise, it can be any valid `option` in `pymupdf.Page.get_text`.
        :param clip: (x0, y0, x1, y1)
        :param small_area: If True, we assume `clip` is a small area (e.g. a table cell). When OCR
                           is on, and it's a small area, whatever results we get are joined into
                           one single text block. This makes sure "George Russell" won't be split
                           into "George" and "Russell". When OCR is off, this parameter has no
                           effect. Default is False
        :param expected: If provided, only texts that match this regex will be returned. Default is
                         return everything w/o filtering. E.g., when we extract texts in the lap
                         time col., we want `expected` to be `re.compile(r'\d+:\d+\.\d+')`.
        :param kwargs: Other keyword arguments to pass to `pymupdf.Page.get_text`
        :return: Same type as the return of `pymupdf.Page.get_text`
        """
        def _return_empty() -> str | list | dict:
            """Helper function to return empty results with the appropriate type"""
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

        def _result_is_empty(r: str | list | dict) -> bool:
            """Check if `.get_text` results are empty"""
            if isinstance(r, str):
                return not r.strip()
            elif isinstance(r, list):
                return not any([i[4].strip() for i in r])
            elif isinstance(r, dict):
                return not any([span['text'].strip()
                                for block in r['blocks']
                                for line in block['lines']
                                for span in line['spans']])
            else:
                raise ValueError(f'Unknown type of results: {type(r)}. Expected str, list, or '
                                 f'dict. Got {r}')

        # Try simple search first
        results = self._pymupdf_page.get_text(option, clip=clip, **kwargs)
        results = self._clean_get_text_results(results, option, expected)
        if not _result_is_empty(results):
            return results

        # If `clip` is not provided, OCR the whole page and use `search_for`. We should always
        # provide `clip` whenever possible, as OCR the whole page would very often mess up the
        # positioning, e.g. "George Russell" may be OCR-ed as "George" and "Russell", respectively.
        if clip is None:
            results = self.ocred_page.get_text(option, clip)
            results = self._clean_get_text_results(results, option, expected)
            if _result_is_empty(results):
                _return_empty()
            else:
                return results

        # If we have `clip`, only OCR the clipped small area
        # First check if any black pixels
        """
        We first check if there are some black pixels (at least 10 pixels with RGB < 50) in the
        clipped area. If not, return empty immediately. This is because all texts are black(ish).
        We check this for two reasons:

        1. it's much faster to skip OCR if we already know there is no text in the clipped area
        2. tesseract quality is really bad. It may say a short light grey line is "-", which breaks
           most of our parsing. So we try our best to avoid OCR-ing such areas
        """
        pixmap = self.get_pixmap(clip=clip, dpi=DPI)
        pixmap_arr = np.ndarray(
            [pixmap.height, pixmap.width, 3], dtype=np.uint8, buffer=pixmap.samples
        ).copy()  # TODO: `samples_mv` memory error on Mac/Windows?
        # `.copy()` is a must. See https://stackoverflow.com/q/39554660/12867291
        if np.sum(pixmap_arr < 50) < 10:  # noqa: PLR2004
            return _return_empty()

        # Replace light grey pixel (RGB > 200) with white. This improves OCR quality a lot
        pixmap_arr[np.all(pixmap_arr >= 200, axis=2)] = 255  # noqa: PLR2004
        # random_filename = uuid.uuid4().hex
        # Image.fromarray(pixmap_arr).save(self.tempdir / f'{random_filename}.png')
        # try:
        #     i = max([int(i.split('.')[0]) for i in os.listdir('training/statis')]) + 1
        # except:
        #     i = 1
        # Image.fromarray(pixmap_arr).save(f'training/labelling/statis/{i}.png')

        # OCR the clipped area
        results = OCR.predict(pixmap_arr)
        if len(results) != 1:
            raise OCRError(f'Unexpected OCR results: {results}')
        elif len(results) == 0:
            return _return_empty()
        results = results[0]
        if small_area:
            text = ' '.join([i.strip() for i in results['rec_texts']]).strip()
            if not text:
                return _return_empty()
            bbox = (min([i[0] for i in results['rec_boxes']]),
                    min([i[1] for i in results['rec_boxes']]),
                    max([i[2] for i in results['rec_boxes']]),
                    max([i[3] for i in results['rec_boxes']]))
            results = [TextBlock(
                text=OCR_ERRORS.get(text, text),
                bbox=self._transform_bbox(
                    original_bbox=clip,
                    new_bbox=(0, 0, pixmap.width, pixmap.height),
                    bbox=bbox
                ))]
        else:
            results = [TextBlock(
                text=OCR_ERRORS.get(i[1], i[1]),
                bbox=self._transform_bbox(
                    original_bbox=clip,
                    new_bbox=(0, 0, pixmap.width, pixmap.height),
                    bbox=tuple(i[0])
                )
            ) for i in zip(results['rec_boxes'], results['rec_texts'])]

        # TODO: finish this regex check
        # if expected and (not re.match(expected, text)):
        #     _return_empty()

        # If code reaches here, we got some non-empty text (either matches `expected` or there is
        # no `expected` provided). Return the text in appropriate type, depending on `option`. See
        # https://pymupdf.readthedocs.io/en/latest/textpage.html
        match option:
            case 'text':
                text = ''.join(
                    [c for c in ' '.join([i.text for i in results]).strip() if c in printable]
                ).strip()
                return text
            case 'words':
                return [
                    (
                        *i.bbox,
                        ''.join([c for c in i.text.strip() if c in printable]),
                        -1,
                        -1,
                        -1
                    ) for i in results
                ]
            case 'blocks':
                return [
                    (
                        *i.bbox,
                        ''.join([c for c in i.text.strip() if c in printable]),
                        -1,
                        -1
                    ) for i in results
                ]
            case 'dict':
                res = {'blocks': []}
                for i in results:
                    res['blocks'].append({
                        'bbox': i.bbox,
                        'lines': [{
                            'bbox': i.bbox,
                            'spans': [{
                                'bbox': i.bbox,
                                'text': ''.join([c for c in i.text.strip() if c in printable])
                            }]
                        }]
                    })
                return res
            case _:
                raise ValueError(f'Native `.get_text` does not find anything. For OCR, `option` '
                                 f'must be one of "text", "words", "blocks", or "dict". Got '
                                 f'"{option}"')

    def has_horizontal_line(
            self,
            clip: tuple[float, float, float, float],
            length: float = 0.5,
            colour: tuple[int, int, int] = (128, 128, 128),
            colour_tol: float = 10
    )-> bool:
        """Check if there is a long enough horizontal line in the given `clip` area

        This method implicitly requires `clip` to be accurate enough. That is, if `clip` has a
        width of 100px, then only a horizontal line with a length of at least `length` * 100px will
        be considered as a horizontal line here. This method is often used to detect the strikeout
        text.

        TODO: combine with `.search_for_black_line`?

        :param clip: (x0, y0, x1, y1)
        :param length: The minimum length of the horizontal line as a share of the width of `clip`.
                       Lines shorter than this are ignored. Default is 0.5, i.e. 50% of the width
        :param colour: RGB colour of the horizontal line. Default is (128, 128, 128). Quali. lap
                       times PDFs use this colour. We allow for +/-10 tolerance for each channel
        :param colour_tol: Tolerance for each channel of the `colour`. Default is 10
        :return: True if found
        """
        # Pixels with the specified colour --> 1; 0 otherwise
        pixmap = self.get_pixmap(clip=clip, dpi=DPI)
        pixmap = np.ndarray(
            [pixmap.height, pixmap.width, 3], dtype=np.uint8, buffer=pixmap.samples
        ).copy()
        pixmap = np.all(np.abs(pixmap - colour) <= colour_tol, axis=2)

        # Drop the first 20% and last 20% rows, to avoid table header or bottom lines. These won't
        # be the horizontal lines we want
        """
        E.g., the zero-th row's top may be the line of the table header. When we get the pixmap of
        this area, this table top line will be included. The line itself is usually black, so won't
        be detected as a grey horizontal line. However, because above this line is white, and
        possibly due to anti-aliasing, the pixels just above this line may be light grey. This can
        lead to false positive detection of horizontal lines. So we drop the first and last 20%
        rows to avoid this problem.
        """
        pixmap = pixmap[int(pixmap.shape[0] * 0.2):int(pixmap.shape[0] * 0.8), :]

        # Check if a row has at least `length` continuous grey pixels
        min_length = int(pixmap.shape[1] * length)
        for row in pixmap:
            count = 0
            for pixel in row:
                if pixel:
                    count += 1
                    if count >= min_length:
                        return True
                else:
                    count = 0
        return False

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

    def search_for_black_line(
            self,
            clip: Optional[tuple[float, float, float, float]] = None,
            max_thickness: float = 2,
            min_length: float = 0.8,
            scaling_factor: float = 4
    ) -> list[float]:
        """Search for a long black horizontal line in the page

        :param clip: (x0, y0, x1, y1). If provided, only search in this area. Otherwise, search the
                     whole page
        :param max_thickness: The max. thickness of the line in pixels. This helps to distinguish
                              black box vs black line. Default is 2px
        :param min_length: The minimum length of the line as a proportion of the `clip` width. The
                           default is 0.8, i.e., the line should span at least 80% of the width of
                           the `clip` area
        :param scaling_factor: Upsample the image by this factor before searching for lines. This
                               helps to detect thin lines. Default is 4, which means 1px line in
                               the original image becomes 4px in the upsampled image
        :return: The y-midpoints of the found lines in a list
        """
        # Get the pixmap of the clipped area
        clip = clip if clip else None
        pixmap = self.get_pixmap(clip=clip,
                                 matrix=pymupdf.Matrix(scaling_factor, 0, 0, scaling_factor, 0, 0))
        l, t, r, b = pixmap.x, pixmap.y, pixmap.x + pixmap.w, pixmap.y + pixmap.h
        pixmap = np.ndarray([b - t, r - l, 3], dtype=np.uint8, buffer=pixmap.samples)

        # Find rows with sufficient contiguous black pixels
        is_black_row = np.mean(pixmap < 50, axis=(1, 2)) > min_length  # noqa: PLR2004

        # Sample down to original resolution. If any row is black, then the downsampled row is also
        # treated as black
        is_black_row = np.array([np.any(is_black_row[i:i + scaling_factor])
                                 for i in range(0, len(is_black_row), scaling_factor)])

        # Find consecutive black rows that are at most `max_thickness`px thickness
        max_thickness *= scaling_factor
        black_lines = []
        line_start = None
        for i, is_black in enumerate(is_black_row):
            if is_black and line_start is None:
                line_start = i
            elif not is_black and line_start is not None:
                if i - line_start <= max_thickness:
                    black_lines.append((i + line_start) / 2)  # Midpoint of the line
                line_start = None

        # Edge case for the line being at the bottom of `clip`. Shouldn't happen but just in case
        if line_start is not None and len(is_black_row) - line_start <= max_thickness:
            black_lines.append((line_start + len(is_black_row)) / 2)

        # Convert to original page coordinates
        if clip:
            black_lines = [i + clip[1] for i in black_lines]
        return black_lines

    def search_for_white_strip(
            self,
            height: float = 4,
            clip: Optional[tuple[float, float, float, float]] = None
    ) -> list[float]:
        """Search for a wide horizontal white strip in the page, with at least `height` px tall

        :param height: The minimum height of the white strip in pixels. Default is 6 px, which is
                       roughly half of the normal line height of the FIA PDFs
        :param clip: (x0, y0, x1, y1). If provided, only search in this area. Otherwise, search the
                     whole page
        :return: The top y-coord. of the found white strips in a list, sorted from top to bottom
        """
        # TODO: this 4px height is very fragile. Better if get the exact vertical gap in pixels
        #       between two car No. and use (some proportion of) this gap as `height`

        # Get the pixmap of the clipped area
        clip = clip if clip else self.bound()
        pixmap = self.get_pixmap(clip=clip)
        l, t, r, b = pixmap.x, pixmap.y, pixmap.x + pixmap.w, pixmap.y + pixmap.h
        pixmap = np.ndarray([b - t, r - l, 3], dtype=np.uint8, buffer=pixmap.samples)

        # Find all white rows
        is_white_row = np.all(pixmap == 255, axis=(1, 2))  # noqa: PLR2004
        white_strips = []
        strip_start = None

        # Find consecutive white rows that are at least `height` px tall
        for i, is_white in enumerate(is_white_row):
            if is_white and strip_start is None:
                strip_start = i
            elif not is_white and strip_start is not None:
                if i - strip_start >= height:
                    white_strips.append(strip_start + t)
                strip_start = None

        # Edge case for the strip being at the bottom
        if strip_start is not None and len(is_white_row) - strip_start >= height:
            white_strips.append(strip_start + t)
        return white_strips

    def search_for_grey_white_rows(
            self,
            clip: Optional[tuple[float, float, float, float]] = None,
            min_height: float = 0,
            min_width: float = 0.8
    ) -> Optional[list[float]]:
        """Search for grey/white rows in the page

        The table rows in FIA docs. are coloured in white and grey background, interleaved. This
        method first locates all grey rows, and then the white rows are in between two consecutive
        grey rows. We also take care of the case where the first and/or the last row are in white
        background. The returned list is the y-coords. of each row, which can be conveniently
        passed to other methods as the `hlines` parameter.

        :param clip: (x0, y0, x1, y1). If provided, only search in this area. Otherwise, search the
                     whole page
        :param min_height: The minimum height of the grey row in pixels. If a grey row is shorter
                           in height than this, it will be ignored. Default is 0, which means no
                           filtering on height
        :param min_width: The minimum width of the grey row as a proportion of the `clip` width.
                          The default is 0.8, i.e., a row with at least 80% grey pixels is treated
                          as a grey row
        :return: A list of numbers, where each number is the top y-coord. of a row. The bottom of
                 the last row is also included
        """
        # Get the pixmap of the clipped area
        clip = clip if clip else self.bound()
        pixmap = self.get_pixmap(clip=clip, dpi=DPI)
        l, t, r, b = pixmap.x, pixmap.y, pixmap.x + pixmap.w, pixmap.y + pixmap.h
        pixmap = np.ndarray([b - t, r - l, 3], dtype=np.uint8, buffer=pixmap.samples)

        # Convert minimum row height to the new DPI
        scaling_factor = pixmap.shape[0] / (clip[3] - clip[1])
        min_height = scaling_factor * min_height

        # Find all grey/white rows. Grey is usually RGB = 232
        is_grey_row = np.mean(pixmap < 240, axis=(1, 2)) > min_width  # noqa: PLR2004
        grey_rows = []
        row_start = None
        for i, is_grey in enumerate(is_grey_row):
            if is_grey and (row_start is None):
                row_start = i
            elif (not is_grey) and (row_start is not None):
                if i - row_start > min_height:
                    grey_rows.append((row_start, i))
                row_start = None
        if row_start and (len(is_grey_row) - row_start > min_height):  # If the last row is grey
            grey_rows.append((row_start, len(is_grey_row)))

        # All rows should have roughly the same height
        row_heights = ([grey_rows[i][1] - grey_rows[i][0] for i in range(len(grey_rows))]
                       + [grey_rows[i][0] - grey_rows[i - 1][1] for i in range(1, len(grey_rows))])
        row_height = np.mean(row_heights) if row_heights else 0
        if not np.all(np.isclose(row_heights, row_height, rtol=0.1)):
            outliers = [i for i in grey_rows if not np.isclose(i[1] - i[0], row_height, rtol=0.2)]
            for row in outliers:
                text = self.get_text('text', clip=(l, clip[1] + row[0] / scaling_factor,
                                                   r, clip[1] + row[1] / scaling_factor))
                if 'antonelli' in text.lower():  # See e4df38f. Basically sometimes FIA wraps
                    continue                     # Antonelli's name into two lines, making the row
                else:                            # taller
                    raise ParsingError(f'Found rows with different heights on p.{self.number} in '
                                       f'{self.file}. Expected all rows to have similar heights: '
                                       f'{row_heights}')
        """
        We use 20% as "roughly the same". It's actually a very tight tolerance. First, when we get
        pixmap of the page, it's a matrix. So we will have coords./indices like (1, 5), not
        (1.1, 4.9). This creates some rounding error. After scaling (DPI = 600), this error is
        further amplified. Second, the `clip` may not always be perfect. The true table area may be
        (0, 10, 100, 500), and `clip` may be (0, 12, 100.6, 499). This may affect the row height of
        the first and/or the last row. Therefore, 20% is a reasonable tolerance.
        """
        # Convert the grey rows to the original page coordinates
        grey_rows = [(clip[1] + i[0] / scaling_factor, clip[1] + i[1] / scaling_factor)
                     for i in grey_rows]
        row_height /= scaling_factor

        # White rows are in between two consecutive grey rows
        white_rows = [(grey_rows[i][1], grey_rows[i + 1][0]) for i in range(len(grey_rows) - 1)]

        # Check if the first row is white. (If it's grey, already captured above)
        # First check if have enough spacing above the first grey row
        t_first_grey_row = grey_rows[0][0] if grey_rows else clip[3]  # May have zero grey row
        if t_first_grey_row > clip[1] + row_height * 0.7:
            # Then check if any text in the area above the first grey row
            text = self.get_text('text', clip=(clip[0], clip[1], clip[2], t_first_grey_row))
            if text.strip():
                # If yes, then it's a white row
                if row_height:
                    """
                    E.g., we only have one row in the table, and it's white. Because there is no
                    grey row, `row_height` will be default to zero above. In this case, we just use
                    the top the `clip` as the top of the (and the only) white row.
                    """
                    white_rows.insert(0, (t_first_grey_row - row_height, t_first_grey_row))
                else:
                    white_rows.insert(0, (clip[1], t_first_grey_row))

        # Check if the last row is white, in the same way
        b_last_grey_row = grey_rows[-1][1] if grey_rows else clip[1]
        if b_last_grey_row < clip[3] - row_height * 0.7:
            text = self.get_text('text', clip=(clip[0], b_last_grey_row, clip[2], clip[3]))
            if text.strip():
                if row_height:
                    white_rows.append((b_last_grey_row, b_last_grey_row + row_height))
                else:
                    white_rows.append((b_last_grey_row, clip[3]))

        # Get the rows' y-coords.
        temp = sorted([coord for row in grey_rows + white_rows for coord in row])
        if not temp:
            return []
        hlines = [temp[0]]
        for i in range(1, len(temp)):
            if not np.isclose(temp[i], temp[i - 1], rtol=0.01):
                hlines.append(temp[i])
        return hlines


class TextBlock:
    """A text block on a PDF page, with its bounding box and text content

    :param bbox: The bounding box of the text block in the format (l, t, r, b)
    :param text: Text
    """
    def __init__(self, bbox: tuple[float, float, float, float], text: str):
        self.text = text
        self.l = self.x0 = bbox[0]
        self.t = self.y0 = bbox[1]
        self.r = self.x1 = bbox[2]
        self.b = self.y1 = bbox[3]
        self.bbox = (self.l, self.t, self.r, self.b)

    def __repr__(self) -> str:
        return (f'TextBlock(text="{self.text}", '
                f'bbox=({self.l:.2f}, {self.t:.2f}, {self.r:.2f}, {self.b:.2f}))')


class ParsingError(Exception):
    pass


class OCRError(Exception):
    pass
