# -*- coding: utf-8 -*-
"""Module for PDF page reading with OCR support"""
import logging
import os
import re
import warnings
from dataclasses import dataclass, field
from functools import cached_property
from types import SimpleNamespace
from typing import Literal, Optional, Sequence
from string import printable

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
import pymupdf
from paddleocr import PaddleOCR

DPI = 600  # Works the best for OCR
OCR = PaddleOCR(lang='en',
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False)

# Common OCR mistakes
# TODO: very fragile...
OCR_ERRORS = {
    'KMIH': 'KM/H',
    'L': '1',
    'I': '1'
}

# Strikeout line location: it shouldn't be too close to the top or bottom of the text bbox
STRIKEOUT_LINE_MARGIN = 0.2


class Page:
    def __init__(self, page: pymupdf.Page, file: str | os.PathLike):
        self._pymupdf_page = page  # Original PyMuPDF page
        self.file = file           # The PDF file path. Mainly for debug purpose
        self.w = page.bound()[2]   # Width and height, for convenience
        self.h = page.bound()[3]

    def __getattr__(self, name: str):
        return getattr(self._pymupdf_page, name)  # If a method/attr. is not found, use PyMuPDF's

    @cached_property
    def ocred_page(self) -> SimpleNamespace:
        """OCR the entire page and return the texts with their bounding boxes

        :return: An object w/ `.get_text` and `.search_for` methods, as if it is a `pymupdf.Page`
        TODO: move to OCR module later
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

    def show_page(self, clip: Optional[tuple[float, float, float, float]] = None) -> None:
        """May not work well depending on screen resolution and/or DPI. For debug process only

        See https://github.com/pymupdf/PyMuPDF-Utilities/blob/master/table-analysis/show_image.py
        """
        pixmap = self.get_pixmap(clip=clip, dpi=DPI)
        img = np.frombuffer(buffer=pixmap.samples_mv, dtype=np.uint8) \
            .reshape((pixmap.height, pixmap.width, 3))
        plt.figure(dpi=DPI)
        plt.imshow(img, extent=(0, pixmap.w * 72 / DPI, pixmap.h * 72 / DPI, 0))
        plt.show()
        return

    def search_for(self, text: str, **kwargs) -> list['TextBlock']:
        """`pymupdf.Page.search_for`, with OCR"""
        results: list[pymupdf.Rect]

        # Usual search
        if results := self._pymupdf_page.search_for(text, **kwargs):
            return [TextBlock(text=text, bbox=(tuple(i))) for i in results]

        # If nothing found, OCR the page and search again
        # TODO
        return self.ocred_page.search_for(text, **kwargs)

    @staticmethod
    def _transform_bbox(
            bbox: tuple[float, float, float, float],
            from_page_bound: tuple[float, float, float, float],
            to_page_bound: tuple[float, float, float, float]
    ) -> tuple[float, float, float, float]:
        """Transform the bbox from new coord. system to the original system

        :param bbox: The bounding box to be transformed
        :param from_page_bound: The bounds of the current PDF page, i.e. the page's (l, t, r, b) of
                                the page where `bbox` is located. This bound does not necessarily
                                need to start from (0, 0). E.g., can be (100, 200, 500, 800)
        :param to_page_bound: The page bounds of the target page

        >>> Page._transform_bbox((10, 20, 30, 40), (0, 0, 100, 200), (0, 0, 200, 400))
        (20.0, 40.0, 60.0, 80.0)
        >>> Page._transform_bbox((10, 20, 30, 40), (0, 0, 300, 200), (10, 20, 40, 40))
        (11.0, 22.0, 13.0, 24.0)
        """
        scale_x = (to_page_bound[2] - to_page_bound[0]) / (from_page_bound[2] - from_page_bound[0])
        scale_y = (to_page_bound[3] - to_page_bound[1]) / (from_page_bound[3] - from_page_bound[1])
        return (to_page_bound[0] + (bbox[0] - from_page_bound[0]) * scale_x,
                to_page_bound[1] + (bbox[1] - from_page_bound[1]) * scale_y,
                to_page_bound[0] + (bbox[2] - from_page_bound[0]) * scale_x,
                to_page_bound[1] + (bbox[3] - from_page_bound[1]) * scale_y)

    def _native_get_text(
            self,
            option: Literal['text', 'words', 'blocks', 'dict'] = 'text',
            **kwargs
    ) -> list['TextBlock']:
        """
        Same as `pymupdf.Page.get_text`, but the return type, no matter whichever `option` is
        specified, is changed to a list of `TextBlock` for easier processing later

        This is highly customised to our use case. For example, we assume that when `option=dict`,
        there can be at most one superscript and at most one regular text in one `.get_text()`
        call, etc. That is, whenever we call `.get_text()` with `option=dict`, we are getting the
        text within a small area, e.g. a table cell
        """
        if option not in {'text', 'words', 'blocks', 'dict'}:
            raise NotImplementedError(f'`option` can only be one of "text", "words", '
                                      f'"blocks", or "dict". Got "{option}"')
        res = self._pymupdf_page.get_text(option=option, **kwargs)

        textblocks: list[TextBlock]
        if option == 'text':
            if text := self._clean_get_text_result(res):
                return [TextBlock(text=text)]
            else:
                return []

        elif option in {'blocks', 'words'}:
            textblocks = []
            for i in res:
                if text := self._clean_get_text_result(i[4]):
                    textblocks.append(TextBlock(text=text, bbox=i[:4]))
            return textblocks

        else:  # option == 'dict'
            # First, clean everything found
            spans = []
            for i in res['blocks']:
                if 'lines' in i:
                    for j in i['lines']:
                        for k in j['spans']:
                            if text := self._clean_get_text_result(k['text']):
                                k['text'] = text
                                spans.append(k)
            if not spans:
                return []

            # When `option = dict`, we need to check if any text is a superscript
            error_message: str = (f'Unable to infer whether a text is a regular text or a '
                                  f'superscript: {res}. Error occurred on p. {self.number} in '
                                  f'{self.file}')
            match len(spans):
                # If only one text found, this should be a usual/non-superscript text
                case 1:
                    textblocks = [TextBlock(text=spans[0]['text'], bbox=spans[0]['bbox'])]
                # Two texts found. Should be one usual text and one superscript
                case 2:
                    n_superscripts: int = 0
                    n_regular_texts: int = 0
                    for span in spans:  # See https://pymupdf.readthedocs.io/en/latest/recipes-text.html#how-to-analyze-font-characteristics  # noqa: E501
                        if span['flags'] == 1:
                            n_superscripts += 1
                            superscript = span
                        else:
                            n_regular_texts += 1
                            regular_text = span
                    """
                    If the PDF is formatted correctly, then we should have one superscript and
                    one regular text, and we are done. However, FIA is stupid and occasionally
                    use a smaller font size for the superscript, but do not set the superscript
                    flag. So if we found two regular texts above, then have to decide which is
                    the superscript using font size (#48). In principle superscript font size
                    should be sufficiently smaller than the regular text. We use 20% diff. in
                    font size as a cutoff
                    """
                    size_cutoff: float = 0.2
                    if (n_superscripts == 2) or (n_regular_texts == 2):   # noqa: PLR2004
                        temp = (spans[0]['size'] + spans[1]['size']) / 2  # Avg. font size
                        if (spans[0]['size'] - spans[1]['size']) / temp > size_cutoff:
                            superscript = spans[1]
                            regular_text = spans[0]
                        elif (spans[1]['size'] - spans[0]['size']) / temp > size_cutoff:
                            superscript = spans[0]
                            regular_text = spans[1]
                        else:
                            raise ParsingError(error_message)
                    textblocks = [
                        TextBlock(text=regular_text['text'], bbox=regular_text['bbox']),
                        TextBlock(text=superscript['text'], bbox=superscript['bbox'],
                                  superscript=True)
                    ]
                # Should never have three or more texts in one `.get_text()` in our parsing
                case _:
                    raise ParsingError(error_message)

            # When `option = dict`, also need to check if any strikeout text
            for textblock in textblocks:
                l, t, r, b = textblock.bbox
                h = b - t
                if self.search_for_black_lines(
                        clip=(l, t + STRIKEOUT_LINE_MARGIN * h, r, b - STRIKEOUT_LINE_MARGIN * h),
                        min_length=0.9,  # We are using the text's bbox, so relative to it, the
                        rgb=192          # strikeout line should span almost the entire width
                ):
                    textblock.strikeout = True
            return textblocks

    @staticmethod
    def _clean_get_text_result(text: str, expected: Optional[re.Pattern] = None) -> Optional[str]:
        r"""
        Remove non-printable chars. from `text`, and return it only if the cleaned `text` regex
        matches `expected`, if `expected` is provided

        :param text: A string
        :param expected: An regex pattern. E.g., when we extract texts in the lap time col.,
                         `expected` would probably be `re.compile(r'\d+:\d+\.\d+')`.
        :return: The cleaned text. None if no match w/ `expected`
        """
        text = ''.join([c for c in text if c in printable]).strip()
        if not text:  # i.e., if `text == ''`, will return `None`
            return None

        if expected:
            if re.match(expected, text):
                return text
            else:
                return None
        return text

    def get_text(
            self,
            option: Literal['text', 'words', 'blocks', 'dict'] = 'text',
            clip: Optional[tuple[float, float, float, float]] = None,
            small_area: bool = False,
            expected: Optional[re.Pattern] = None,
            **kwargs
    ) -> list['TextBlock']:
        r"""This is `pymupdf.Page.get_text` w/ OCR functionality

        This method does the native `.get_text` first. If no text that regex matches `expected` is
        found (, suppose `expected` is provided), we proceed with OCR.

        * `clip` is not None: only OCR the clipped area instead of the whole page, so it's faster
        * `clip` is None: OCR the whole page and search for `text`

        :param option: Unlike original `pymupdf.Page.get_text`, we only support `option` to be
                       "text", "words", "blocks", or "dict"
        :param clip: (x0, y0, x1, y1)
        :param small_area: If True, we assume `clip` is a small area (e.g. a table cell). When OCR
                           is on, and it's a small area, whatever results we get are joined into
                           one single text block. This makes sure text like "George Russell" won't
                           be split into two strings "George" and "Russell". When OCR is off, this
                           parameter has no effect. Default is False
        :param expected: If provided, only texts that match this regex will be returned. Default is
                         return everything w/o filtering. E.g., when we extract texts in the lap
                         time col., we want `expected` to be `re.compile(r'\d+:\d+\.\d+')`.
        :param kwargs: Other keyword arguments to pass to `pymupdf.Page.get_text`
        :return: A list of `TextBlock`s
        """
        if expected is not None:
            raise NotImplementedError('`expected` is not supported yet')

        # Try simple search first
        if results := self._native_get_text(option, clip=clip, **kwargs):
            return results
        logging.debug('No text found natively. Proceed with OCR')

        # TODO: will fail if `clip` has partial usual text and partial image text...

        # If `clip` is not provided, OCR the whole page and use `search_for`. We should always
        # provide `clip` whenever possible, as OCR the whole page would very often mess up the
        # positioning, e.g. "George Russell" may be OCR-ed as "George" and "Russell". And it's very
        # slow
        if clip is None:
            if results := self.ocred_page.get_text(option, clip):
                return results
            else:
                return []

        # If we have `clip`, only OCR the clipped area
        # First check if any black pixels
        """
        We first check if there are some black pixels (at least 10 pixels with RGB < 50) in the
        clipped area. If not, return empty immediately. This is because all texts are black(ish).
        We check this for two reasons:

        1. it's much faster to skip OCR if we already know there is no text in the clipped area
        2. OCR quality can be really bad. It may say a short light grey line is "-", which breaks
           most of our parsing. So we try our best to avoid OCR-ing such areas
        """
        pixmap: pymupdf.Pixmap = self.get_pixmap(clip=clip, dpi=DPI, annots=False)
        pixmap_arr: npt.NDArray[np.uint8] = (np.frombuffer(buffer=pixmap.samples_mv,
                                                           dtype=np.uint8)
                                             .reshape((pixmap.height, pixmap.width, 3))
                                             .copy())
        """
        To get `pixmap` into a numpy array of RGB pixels, we can use either `pixmap.samples` or
        `pixmap.samples_mv`. The difference is that `pixmap.samples` returns a bytes object, i.e. a
        copy of the original pixmap, while `pixmap.samples_mv` is a memory view/pointer to the
        original pixmap. So the latter is significantly faster.
        
        However, we still must create a deep `.copy()`. Because PaddleOCR may process the input
        image inplace, and we are working with a memory view, so during the OCR, the original
        pixmap may be changed, leading to unexpected errors. I sometimes/very often get segfault
        errors without the `.copy()`, but can't reproduce it consistently. For safety, always do a
        copy here.
        
        A quick benchmark shows that using `pixmap.samples_mv` + `.copy()` is 5x faster than using
        `pixmap.samples`. This is why we reach the above code.
        """
        if np.sum(pixmap_arr < 50) < 10:  # noqa: PLR2004
            return []

        # Replace light grey pixel (RGB > 200) with white. This improves OCR quality a lot
        pixmap_arr[np.all(pixmap_arr >= 200, axis=2)] = 255  # noqa: PLR2004

        # OCR the clipped area
        match OCR.predict(pixmap_arr):
            case []:
                return []
            case [dict() as ocr_results]:
                if small_area:
                    text: str = ' '.join([self._clean_get_text_result(i)
                                          for i in ocr_results['rec_texts']])
                    if not text:
                        return []
                    bbox: tuple[float, float, float, float] = (
                        min([i[0] for i in ocr_results['rec_boxes']]),
                        min([i[1] for i in ocr_results['rec_boxes']]),
                        max([i[2] for i in ocr_results['rec_boxes']]),
                        max([i[3] for i in ocr_results['rec_boxes']])
                    )
                    """
                    The `bbox` here is the bbox in the pixmap coord. system, i.e. the original page
                    after DPI scaling, so need to convert it back to the original page coord.
                    """
                    textblocks: list[TextBlock] = [TextBlock(
                        text=OCR_ERRORS.get(text, text),
                        bbox=self._transform_bbox(
                            bbox=bbox,
                            from_page_bound=(0, 0, pixmap.width, pixmap.height),
                            to_page_bound=clip
                        ))]
                else:
                    textblocks = []
                    for i in zip(ocr_results['rec_boxes'], ocr_results['rec_texts']):
                        cleaned_text: str = self._clean_get_text_result(i[1])
                        cleaned_text = OCR_ERRORS.get(cleaned_text, cleaned_text)
                        if not cleaned_text:
                            continue
                        textblocks.append(TextBlock(
                            text=cleaned_text,
                            bbox=self._transform_bbox(
                                bbox=tuple(i[0]),
                                from_page_bound=(0, 0, pixmap.width, pixmap.height),
                                to_page_bound=clip
                            ))
                        )
            case other:
                raise OCRError(f'Unexpected OCR results: {other}')

        # TODO: finish this regex check
        # if expected and (not re.match(expected, text)):
        #     _return_empty()
        return textblocks

    def search_for_black_lines(
            self,
            clip: Optional[tuple[float, float, float, float]] = None,
            max_thickness: float = 2,
            min_length: float = 0.8,
            scaling_factor: int = 4,
            rgb: float = 50
    ) -> list[float]:
        """Search for long black horizontal lines in the `clip` area and return their y-coords.

        :param clip: (x0, y0, x1, y1). If provided, only search in this area. Otherwise, search the
                     whole page
        :param max_thickness: The max. thickness of the line in pixels. This helps to distinguish
                              rectangles filled in black vs black line. Default is 2px. This is
                              very fragile...
        :param min_length: The minimum length of the line as a proportion of the `clip` width. The
                           default is 0.8, i.e., the line should span at least 80% of the width of
                           the `clip` area
        :param scaling_factor: Upsample the `clip` area by this factor before searching for lines.
                               This helps to detect thin lines. Default is 4, which means 1px line
                               in the original image becomes 4px in the upsampled image
        :param rgb: The max. RGB value for a pixel to be considered as black. Default is 50. If we
                    want to find grey lines as well, can increase this value
        :return: The y-coords. of the found lines as a list. If found multiple, will be sorted in
                 from top to bottom
        """
        # Get the pixmap of the clipped area
        pixmap: pymupdf.Pixmap = self.get_pixmap(
            clip=clip,
            matrix=pymupdf.Matrix(scaling_factor, 0, 0, scaling_factor, 0, 0)
        )
        pixmap_arr: npt.NDArray[np.uint8] = (np.frombuffer(buffer=pixmap.samples_mv,
                                                           dtype=np.uint8)
                                             .reshape((pixmap.height, pixmap.width, 3))
                                             .copy())

        # Find rows with sufficiently many contiguous black pixels
        n_rows, n_cols, _ = pixmap_arr.shape
        # Whether a pixel is black (RGB < `rgb`). Shape = (n_rows, n_cols)
        is_black_pixel: npt.NDArray[np.bool_] = np.all(pixmap_arr < rgb, axis=2)
        # Add white border to left and right to simplify diff calculation later
        # Shape = (n_rows, n_cols + 2)
        padded: npt.NDArray[np.bool_] = np.pad(is_black_pixel,
                                               ((0, 0), (1, 1)),
                                               mode='constant',
                                               constant_values=False)
        # Whether a pixel has a white or black pixel to the left of it
        # Shape = (n_rows, n_cols + 1)
        # 1  = left pixel is white, this pixel is black --> start of a black run
        # -1 = left pixel is black, this pixel is white --> end of a black run
        # 0  = same colour as left pixel                --> inside a run or all white pixels
        diff: npt.NDArray[np.int8] = np.diff(padded.astype(np.int8), axis=1)
        # Where black runs start and end. Shape = tuple of two arrays, first array is row indices,
        # second array is col. indices
        run_starts: tuple[npt.NDArray[np.intp], npt.NDArray[np.intp]] = np.where(diff == 1)
        run_ends: tuple[npt.NDArray[np.intp], npt.NDArray[np.intp]] = np.where(diff == -1)
        # Length of each black run. Shape = (n_black_runs,)
        run_length: npt.NDArray[np.intp] = run_ends[1] - run_starts[1]
        # Max. black run length per row. Shape = (n_rows,)
        longest_run_per_row: npt.NDArray[np.intp] = np.zeros(n_rows, dtype=np.intp)
        np.maximum.at(longest_run_per_row, run_starts[0], run_length)
        # Whether a row is a black row (i.e., has a sufficiently long black run)
        is_black_row: npt.NDArray[np.bool_] = (longest_run_per_row >= pixmap.width * min_length)

        # Sample down to original resolution
        """
        We use very aggressive down sampling. E.g., w/ `scaling_factor = 4`, a row in the original
        PDF becomes four rows in the upsampled pixmap. If any of these four rows is detected as
        a black row, then we consider the original row as a black row. This is to address grey-ish
        rows due to anti-aliasing. E.g., a very thin black line may become a grey line in the
        original pixmap, and thus may not be detected as a black row. But after upsampling, at
        least one of the four rows should be black enough to be detected as a black row. This
        allows us to detect very thin black lines in the original pixmap.
        
        There may be a small one-off error here. If the #. of rows in the pixmap is not perfectly
        divisible by `scaling_factor`, then the last few rows may be ignored. But this should
        little impact: we don't expect any black lines to be at the bottom of `clip` area.
        """
        is_black_row = np.any(
            is_black_row[:n_rows - (n_rows % scaling_factor)].reshape(-1, scaling_factor),
            axis=1
        )

        # Find consecutive black rows that are at most `max_thickness`px thickness
        diff = np.diff(np.concatenate([[0], is_black_row, [0]]).astype(np.int8))
        # 1: start of a black line; -1: end of a black line
        line_starts: npt.NDArray[np.intp] = np.where(diff == 1)[0]
        line_ends: npt.NDArray[np.intp] = np.where(diff == -1)[0]
        thickness: npt.NDArray[np.intp] = line_ends - line_starts
        valid_lines: npt.NDArray[np.bool_] = (thickness <= max_thickness)
        black_lines: list[float] = (line_starts[valid_lines]
                                    + thickness[valid_lines] / 2
                                    + clip[1] if clip else 0).tolist()
        return sorted(black_lines)  # TODO: need sort? Or already sorted?

    def parse_table_by_grid(
            self,
            vlines: list[float],
            hlines: list[float],
            tol: float = 2,
            allow_multiple_texts_per_cell: Optional[Sequence[int]] = None,
            header_included: bool = True
    ) -> pd.DataFrame:
        """Parse a table cell by cell, defined by lines separating the cols. and rows

        PyMuPDF does have a built-in table parser, but it's bad for many reasons:

        1. it does not always respect the `clip` parameter: sometimes it gets table header/rows
           outside the `clip` area
        2. we can't force it to ignore the table header. E.g., for some short table (e.g. 2 rows),
           it may get the first row as the header, and the second row as the data. However, we
           sometimes want to treat all rows as data, not header. There is no way to force this
        3. sometimes it will get an empty column, if the horizontal gap between two columns are too
           big. This is true even if we manually specify `vertical_lines`

        For these reasons, we always parse tables w/ this method. It returns the table in a usual
        `pd.DataFrame`, and each element/cell is a TextBlock

        :param vlines: x-coords. of vertical lines separating the cols. Table left and right
                       boundaries need to be included
        :param hlines: y-coords. of horizontal lines separating the rows. Table top and bottom
                       boundaries need to be included
        :param tol: tolerance for text and cell positioning. In principle, all texts should fall
                    inside the cell's bounding box. Default is 2 pixels, i.e. if text is within 2px
                    of the cell's boundary, it is considered to be inside the cell. If we find any
                    text more than 2px always from the cell's bbox, will raise an error. See #33
        :param allow_multiple_texts_per_cell: Which cols. can have multiple texts per cell. By
                                              default, we only allow one textblock in one cell.
                                              However, some cases (e.g. reserve drivers in entry
                                              list PDF) may have two textblocks in the zero-th col.
                                              (one normal text and one superscript). In this case,
                                              set this parameter to [0], i.e. allow col. 0 to have
                                              multiple textblocks per cell. Col. indexes start from
                                              0
        :param header_included: whether the first row is header/col. names. Default is False, i.e.
                                treat everything as table content, and no table header/col. names
        :return: A `pd.DataFrame`, with each cell being a TextBlock or a list of TextBlocks,
                 depending on `allow_multiple_texts_per_cell`
        """
        page_no_str = f'p.{self.number} in {self.file}'  # For error/warning messages

        cells: list[list[Optional[TextBlock]]] = []
        for i in range(len(hlines) - 1):
            row = []
            for j in range(len(vlines) - 1):
                l, t, r, b = vlines[j], hlines[i], vlines[j + 1], hlines[i + 1]
                cell_bbox_str = f'({l:.1f}, {t:.1f}, {r:.1f}, {b:.1f})'  # For error/warnings

                # Get text inside the cell defined by (l, t, r, b)
                textblocks = self.get_text('dict', clip=(l, t, r, b))
                if not textblocks:
                    # OK to have an empty cell, e.g. the pit col. in quali. lap times PDF should be
                    # empty for non-stop laps
                    row.append(TextBlock(text=''))
                    continue

                # Need to check if the found text is indeed in the cell's bbox. PyMuPDF is
                # notoriously bad for NOT respecting the `clip` parameter
                for k in range(len(textblocks) - 1, -1, -1):  # TODO: bad practice
                    bbox = textblocks[k].bbox
                    # Drop text outside the cell bbox
                    if (bbox[0] < l - tol) or (bbox[1] < t - tol) or (bbox[2] > r + tol) \
                            or (bbox[3] > b + tol):
                        warnings.warn(f'Text found outside the cell bbox on {page_no_str}. Cell '
                                      f'bbox = {cell_bbox_str}. Text found = {textblocks[k]}. '
                                      f'Ignoring it')
                        textblocks.pop(k)

                # If we had something before, but after filtering out the out-of-bbox texts, now
                # nothing is left, this is an error
                if not textblocks:
                    raise ParsingError(f'Found text outside the cell bbox on {page_no_str}, and '
                                       f'no valid text left after removing them. Row #. = {i}, '
                                       f'col. #. = {j}. Cell bbox = {cell_bbox_str}')

                # Should only have one textblock in a cell, unless `allow_multiple_texts_per_cell`
                # includes this col.
                if allow_multiple_texts_per_cell and (j in allow_multiple_texts_per_cell):
                    row.append(textblocks)
                    continue
                # If this col. is not in `allow_multiple_texts_per_cell`, but we found multiple
                # texts, raise an error
                if len(textblocks) > 1:
                    raise ParsingError(f'Found multiple texts in one cell on {page_no_str}. Row '
                                       f'#. = {i}, col. #. = {j}. Cell bbox = {cell_bbox_str}. '
                                       f'Texts found = {textblocks}')
                # Normal case will reach here
                row.append(textblocks[0])
            cells.append(row)

        if header_included:
            cols: list[TextBlock | list[TextBlock]] = cells.pop(0)
            for i, col_name in enumerate(cols):
                if isinstance(col_name, list):  # In case we allow some cols. to have multiple
                    if len(col_name) > 1:       # texts, the col. name should always be one text
                        raise ParsingError(
                            f'Found multiple texts ina cell in table header on {page_no_str}: '
                            f'{col_name}. Table bbox = ({vlines[0]:.1f}, {hlines[0]:.1f}, '
                            f'{vlines[-1]:.1f}, {hlines[-1]:.1f})'
                        )
                    col_name = col_name[0]
                if col_name is None:
                    # Can be empty, e.g. pit col. in quali. lap times PDF has no col name.
                    cols[i] = TextBlock(text='')
                else:
                    if col_name.superscript or col_name.strikeout:
                        raise ParsingError(
                            f'Found superscript or strikeout text in table header on '
                            f'{page_no_str}: {col_name}. Table bbox = ({vlines[0]:.1f}, '
                            f'{hlines[0]:.1f}, {vlines[-1]:.1f}, {hlines[-1]:.1f})'
                        )
            return pd.DataFrame(cells, columns=[i.text for i in cols], index=None)
        return pd.DataFrame(cells, columns=None, index=None)

    def search_for_white_strips(
            self,
            clip: Optional[tuple[float, float, float, float]] = None,
            height: float = 4
    ) -> list[float]:
        """Search for wide horizontal white strips in the page, with at least `height` px tall

        :param clip: (x0, y0, x1, y1). If provided, only search in this area. Otherwise, search the
                     whole page
        :param height: The minimum height of the white strip in pixels. Default is 4 px, which is
                       roughly half of the normal line height of the FIA PDFs
        :return: The top y-coord. of the found white strips in a list, sorted from top to bottom
        """
        # TODO: this 4px height is very fragile. Better if get the exact vertical gap in pixels
        #       between two car No. and use (some proportion of) this gap as `height`

        # Get the pixmap of the clipped area
        pixmap: pymupdf.Pixmap = self.get_pixmap(clip=clip)
        pixmap_arr: npt.NDArray[np.uint8] = (np.frombuffer(buffer=pixmap.samples_mv,
                                                           dtype=np.uint8)
                                             .reshape((pixmap.height, pixmap.width, 3)))

        # Find all white rows, i.e. strictly all pixels in the row are white (RGB = 255)
        is_white: npt.NDArray[np.bool_] = np.all(pixmap_arr == 255, axis=(1, 2))  # noqa: PLR2004

        # Pad with a black row on top and bottom to simplify. diff calculation later
        padded: npt.NDArray[np.bool_] = np.concatenate([[False], is_white, [False]])

        # Find consecutive white rows
        # Row colour = B B W W W W W W  B W W  B B B W  B
        # diff       =   0 1 0 0 0 0 0 -1 1 0 -1 0 0 1 -1
        # 1: start of a white strip; -1: end of a white strip
        diff: npt.NDArray[np.int8] = np.diff(padded.astype(np.int8))
        starts: npt.NDArray[np.intp] = np.where(diff == 1)[0]
        ends: npt.NDArray[np.intp] = np.where(diff == -1)[0]

        # Filter out white strips that are less than `height` px tall
        strip_heights: npt.NDArray[np.intp] = ends - starts
        is_valid_strip: npt.NDArray[np.bool_] = (strip_heights > height)
        return (starts[is_valid_strip] + pixmap.y).tolist()

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
        pixmap: pymupdf.Pixmap = self.get_pixmap(clip=clip, dpi=DPI)
        pixmap_arr: npt.NDArray[np.uint8] = (np.frombuffer(buffer=pixmap.samples_mv,
                                                           dtype=np.uint8)
                                             .reshape((pixmap.height, pixmap.width, 3))
                                             .copy())

        # Convert minimum row height requirement to the new DPI. E.g., if we scale the page from
        # 300 * 300 to 600 * 600, then 10px in the original page should become 20px
        scaling_factor: float = pixmap_arr.shape[0] / (clip[3] - clip[1])
        min_height_scaled: float = min_height * scaling_factor

        # Find all grey/white rows. Grey is usually RGB = 232
        grey = 240  # TODO: probably make this adjustable later
        is_grey_row: npt.NDArray[np.bool_] = (np.mean(pixmap_arr < grey, axis=(1, 2)) > min_width)

        # Pad with False (non-grey) rows to simplify diff calculation
        padded: npt.NDArray[np.bool_] = np.concatenate([[False], is_grey_row, [False]])

        # Find consecutive grey rows: 1 = start of a grey row; -1 = end of a grey row
        diff: npt.NDArray[np.int8] = np.diff(padded.astype(np.int8))
        starts: npt.NDArray[np.intp] = np.where(diff == 1)[0]
        ends: npt.NDArray[np.intp] = np.where(diff == -1)[0]

        # Filter out grey rows shorter than `min_height`
        row_heights: npt.NDArray[np.intp] = ends - starts
        valid_rows: npt.NDArray[np.bool_] = (row_heights > min_height_scaled)
        # [(row start y-coord., row end y-coord.), ...]
        grey_rows: list[tuple[float, float]] = list(zip(starts[valid_rows].tolist(),
                                                        ends[valid_rows].tolist()))

        # All rows should have roughly the same height
        avg_row_height: float = 0
        if grey_rows:  # OK to have zero grey row, e.g. only one white row in the table
            row_heights = np.array([grey_rows[i][1] - grey_rows[i][0]
                                    for i in range(len(grey_rows))])
            avg_row_height = np.mean(row_heights)
            """
            We use 20% as "roughly the same". It's actually a very tight tolerance. First, when we
            get pixmap of the page, it's a matrix. So we will have coords./indices like (1, 5), not
            (1.1, 4.9). This creates some rounding error. After scaling (DPI = 600), this error is
            further amplified. Second, the `clip` may not always be perfect. The true table area
            may be (0, 10, 100, 500), and `clip` may be (0, 12, 100.6, 499). This may affect the
            row height of the first and/or the last row. Therefore, 20% is a reasonable tolerance
            """

            if not np.all(np.isclose(row_heights, avg_row_height, rtol=0.2)):
                outliers: list[tuple[float, float]] = [
                    i for i in grey_rows if not np.isclose(i[1] - i[0], avg_row_height, rtol=0.2)
                ]
                for row in outliers:
                    text = self.get_text('text', clip=(clip[0], clip[1] + row[0] / scaling_factor,
                                                       clip[2], clip[1] + row[1] / scaling_factor))
                    # See e4df38f. Sometimes FIA wraps Antonelli's name into two lines, making the
                    # row exceptionally taller
                    if any('antonelli' in i.text.lower() for i in text):
                        continue
                    else:
                        raise ParsingError(
                            f'Found rows with different heights inside ({clip[0]:.1f}, '
                            f'{clip[1]:.1f}, {clip[2]:.1f}, {clip[3]:.1f}) on p.{self.number} in '
                            f'{self.file}. Expected all rows to have similar heights'
                        )

            # Convert the grey rows to the original page coordinates
            grey_rows = [(clip[1] + i[0] / scaling_factor, clip[1] + i[1] / scaling_factor)
                         for i in grey_rows]
            avg_row_height /= scaling_factor

        # White rows are in between two consecutive grey rows
        white_rows: list[tuple[float, float]] = [(grey_rows[i][1], grey_rows[i + 1][0])
                                                 for i in range(len(grey_rows) - 1)]

        # Check if the first row is white. (If it's grey, then should already be captured above)
        # First check if have enough spacing (80% of a row height) above the first grey row
        t_first_grey_row = grey_rows[0][0] if grey_rows else clip[3]  # May have zero grey row
        if t_first_grey_row > clip[1] + avg_row_height * 0.8:
            # Then check if any text in the area above the first grey row
            text = self.get_text('text', clip=(clip[0], clip[1], clip[2], t_first_grey_row))
            if any(i.text for i in text):
                # If yes, then it's a white row. Set its row height to avg. row height
                if avg_row_height:
                    white_rows.insert(0, (t_first_grey_row - avg_row_height, t_first_grey_row))
                else:
                    """
                    Sometimes we only have one white row in the table. Because there is no grey
                    row, `avg_row_height` will be default to zero above. In this case, we just use
                    the top the `clip` as the top of the (and the only) white row.
                    """
                    white_rows.insert(0, (clip[1], t_first_grey_row))

        # Check if the last row is white, in the same way
        b_last_grey_row = grey_rows[-1][1] if grey_rows else clip[1]
        if b_last_grey_row < clip[3] - avg_row_height * 0.8:
            text = self.get_text('text', clip=(clip[0], b_last_grey_row, clip[2], clip[3]))
            if any(i.text for i in text):
                if avg_row_height:
                    white_rows.append((b_last_grey_row, b_last_grey_row + avg_row_height))
                else:
                    white_rows.append((b_last_grey_row, clip[3]))

        # Get all rows' start and end y-coords.
        all_rows: list[float] = sorted([coord for row in grey_rows + white_rows for coord in row])
        if not all_rows:
            return []

        # Return the y-coords. where a grey row ends and a white row starts
        """
        E.g., grey rows may be [(10, 29.8), (50, 70)], white rows may be [(30, 50), (70, 90)]. Then
        `all_rows` = [10, 29.8, 30, 50, 50, 70, 70, 90]. We only want to return [10, 29.8 (or 30),
        50, 70, 90]
        """
        hlines: list[float] = [all_rows[0]]
        for i in range(1, len(all_rows)):
            if not np.isclose(all_rows[i], all_rows[i - 1], rtol=0.01):
                hlines.append(all_rows[i])
        return hlines


@dataclass
class TextBlock:
    """A text block, with its text, (if any) bounding box, superscripts, and strikeout texts

    :param text: Text
    :param bbox: The bounding box of the text block in the format (l, t, r, b). If not provided,
                 defaults to (-1, -1, -1, -1)
    :param superscript: A list of superscript texts as strings
    :param strikeout: A list of strikeout texts as strings
    """
    text: str
    bbox: Optional[tuple[float, float, float, float]] = field(default_factory=list)
    l: Optional[float] = field(init=False, default=None)
    t: Optional[float] = field(init=False, default=None)
    r: Optional[float] = field(init=False, default=None)
    b: Optional[float] = field(init=False, default=None)
    x0: Optional[float] = field(init=False, default=None)
    y0: Optional[float] = field(init=False, default=None)
    x1: Optional[float] = field(init=False, default=None)
    y1: Optional[float] = field(init=False, default=None)
    superscript: Optional[bool] = field(default=False)
    strikeout: Optional[bool] = field(default=False)

    def __post_init__(self):
        if not isinstance(self.text, str):
            raise ValueError(f'Invalid `text`: {self.text}. Expected a string')

        if self.bbox is None:  # In case sometimes we pass in `bbox=None`
            self.bbox = []
        if self.bbox:
            if len(self.bbox) != 4:
                raise ValueError(f'Invalid `bbox`: {self.bbox}. Expected a tuple of four floats '
                                 f'representing (l, t, r, b)')
            for i in self.bbox:
                if i < 0:
                    raise ValueError(f'Invalid `bbox`: {self.bbox}. All values in bbox must be '
                                     f'non-negative')
            self.l, self.t, self.r, self.b = self.bbox
            self.x0, self.y0, self.x1, self.y1 = self.bbox

        if self.superscript is None:
            self.superscript = False
        if not isinstance(self.superscript, bool):
            raise ValueError(f'Invalid `superscript`: {self.superscript}. Expected either True or '
                             f'False')
        if self.strikeout is None:
            self.strikeout = False
        if not isinstance(self.strikeout, bool):
            raise ValueError(f'Invalid `strikeout`: {self.strikeout}. Expected either True or '
                             f'False')

    def __repr__(self) -> str:
        ret = f'TextBlock(text="{self.text}"'
        if self.bbox:
            ret += f', bbox=({self.l:.2f}, {self.t:.2f}, {self.r:.2f}, {self.b:.2f})'
        if self.superscript:
            ret += f', superscript=True'
        if self.strikeout:
            ret += f', strikeout=True'
        ret += ')'
        return ret

    def __eq__(self, other):
        if not isinstance(other, TextBlock):
            return False

        if self.text != other.text:
            return False
        if self.superscript != other.superscript:
            return False
        if self.strikeout != other.strikeout:
            return False

        # Compare bbox with tolerance of at most 1px difference
        if self.bbox and other.bbox:  # Both have bbox
            for v1, v2 in zip(self.bbox, other.bbox):
                if abs(v1 - v2) > 1:
                    return False
        elif self.bbox != other.bbox:  # One has bbox and the other doesn't
            return False
        return True


class ParsingError(Exception):
    pass


class OCRError(Exception):
    pass


if __name__ == '__main__':
    pass
