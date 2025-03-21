# -*- coding: utf-8 -*-
import os
import warnings
from pathlib import Path
from string import printable
import tempfile
from typing import Literal, Optional

import numpy as np
import pandas as pd
import pymupdf
import pytesseract


class Page:
    def __init__(self, page: pymupdf.Page):
        self._pymupdf_page = page
        self.ocr_page = None
        self.drawings = page.get_drawings()
        self.crossed_out_text = self.get_strikeout_text()

    def __getattr__(self, name: str):
        return getattr(self._pymupdf_page, name)

    def save_ocr(self, path: Path = Path(tempfile.mkdtemp('fiadoc'))) -> Path:
        """Save the OCR-ed page to a PDF file

        :param path: Folder to save the file. Default is a temporary directory
        :return: Path to the saved file
        """
        if self.ocr_page is None:
            self.get_pixmap(dpi=300).pil_save(path / 'page.png')
            with open(path / 'page.pdf', 'wb') as f:
                f.write(pytesseract.image_to_pdf_or_hocr(str(path / 'page.png'),
                                                         config='--dpi 300'))
            self.ocr_page = pymupdf.open(path / 'page.pdf')[0]  # Memory safe?
        return path / 'page.pdf'

    def search_for_header(self, keyword: str) -> Optional[pymupdf.Rect]:
        """Return the page header location (like "Qualifying Session Final Classification")

        We try to locate the header by:

        1. simple search for the keyword "Final Classification"
        2. if not found, OCR the page, and search again for the same keyword

        :param keyword: Page header text, e.g. "Provisional Classification"
        """
        # Simple keyword search
        found = []
        for text in self.get_text('blocks'):
            if keyword in text[4]:
                found.append(text)
        if len(found) >= 2:
            raise ParsingError(f'Found more than one "{keyword}" on page {self.number + 1} in '
                               f'{self.parent.name}')
        elif len(found) == 1:
            return pymupdf.Rect(found[0][:4])

        # If not found, OCR the page
        if self.ocr_page is None:
            self.save_ocr()

        # Search the OCR-ed page
        for text in self.ocr_page.get_text('blocks'):
            if keyword in text[4]:
                found.append(text)
        if len(found) >= 2:
            raise ParsingError(f'Found more than one "{keyword}" on OCR-ed page '
                               f'{self.number + 1} in {self.parent.name}')

        # If found, need to convert the coords. in the OCR-ed page to the original page
        elif len(found) == 1:
            return pymupdf.Rect(found[0][:4])
        return None

    def search_for(self, text: str, **kwargs) -> list[pymupdf.Rect]:
        """`pymupdf.Page.search_for`, with OCR"""
        # Usual search
        if results := self._pymupdf_page.search_for(text, **kwargs):
            return results

        # If nothing found, OCR the page and search again
        if self.ocr_page is None:
            self.save_ocr()
        return self.ocr_page.search_for(text, **kwargs)

    def get_text(
            self,
            option: str = 'text',
            clip: Optional[tuple[float, float, float, float]] = None,
            **kwargs
    ) -> str | list | dict | None:
        """`pymupdf.Page.get_text`, with OCR

        The OCR is only used if we find nothing in the original page. By "find nothing" we mean no
        "visible" text. E.g., in 2025 Australian quali. final classification PDF, the zero-th col.
        sometimes contains "\x15". Such text can be found by `get_text`, but it's not actually a
        text. So only we care about the "visible" text.

        When `clip` is specified, we
        only OCR the clipped area instead of the whole page. And we assume the clipped area is
        small and contains only a single line of text. That is, we use `--psm 7` in tessaract. See
        https://github.com/tesseract-ocr/tesseract/blob/main/doc/tesseract.1.asc#options.

        When `clip` is not provided, we OCR the whole page. Note that after OCR, the text can be
        slightly off from the original text. E.g. a letter that is originally located in coord.
        (10, 20) can be in (11, 19) after OCR.

        :param option: When `clip` is specified, `option` can only be "text", "words", or "blocks"
        """
        # Try simple search first
        if text := self._pymupdf_page.get_text(option, clip=clip, **kwargs):
            if isinstance(text, list):
                temp = [''.join([c for i in text for c in i[4] if c in printable]).strip()]
                if any(temp):
                    return text
            elif isinstance(text, str):
                temp = ''.join([c for c in text if c in printable]).strip()
                if any(temp):
                    return temp
            elif isinstance(text, dict):
                # warnings.warn('OCR is not supported for `option` being "dict". Returning the '
                #               'original pymupdf result')
                return text

        # If `clip` is not provided, we OCR the whole page
        if clip is None:
            if self.ocr_page is None:
                self.save_ocr()
            return self.ocr_page.get_text(option, **kwargs)

        # If we have `clip`, only OCR the clipped area. First check if any black pixels
        """
        We first check if there is any black pixel in the clipped area. If not, return an empty
        string immediately. This is because all texts are black. We check this for two reasons:
        
        1. it's much faster to skip OCR if we know there is no text in the clipped area
        2. tessaract quality is really bad. If you give it a all grey pure colour image, it can
           spit out some random text like "-". This breaks most of our parsing
           
        TODO: fine-tune tesseract later
        """
        # tempdir = Path(tempfile.mkdtemp('fiadoc'))
        tempdir = Path('temp')
        pixmap = self.get_pixmap(clip=clip, dpi=300)
        pixmap_arr = np.ndarray([pixmap.height, pixmap.width, 3], dtype=np.uint8,
                                buffer=pixmap.samples_mv)
        if not np.any(np.all(pixmap_arr <= 50, axis=2)):
            if option == 'text':
                return ''
            elif option == 'words':
                return []
            elif option == 'blocks':
                return []
            else:
                raise NotImplementedError(f'`option` can only be "text", "words", or "blocks",'
                                          f'when `clip` is specified')

        # If there are some black pixels, OCR the clipped area
        # pixmap.pil_save(tempdir / 'clip.png')
        if os.listdir(tempdir):
            i = max([int(i.removesuffix('.png')) for i in os.listdir(tempdir) if i.endswith('.png')]) + 1
        else:
            i = 0
        pixmap.pil_save(tempdir / f'{i}.png')
        # text = pytesseract.image_to_string(str(tempdir / 'clip.png'), config='--psm 7 --dpi 300')
        text = pytesseract.image_to_string(str(tempdir / f'{i}.png'), config='--psm 7 --dpi 300')
        with open('text.csv', 'a+') as f:
            f.write(f'{i},{text}\n')

        if option == 'text':
            return text
        elif option == 'words':
            if text:
                return [(*clip, text, -1, -1, -1)]
        elif option == 'blocks':
            if text:
                return [(*clip, text, -1, -1)]
        else:
            raise NotImplementedError(f'`option` can only be "text", "words", or "blocks", when '
                                      f'`clip` is specified')

    def get_drawings_in_bbox(self, bbox: tuple[float, float, float, float], tol: float = 1) \
            -> list:
        """Get all drawings in the given bounding box

        :param bbox: (x0, y0, x1, y1)
        :param tol: tolerance in pixels. If a drawing is outside of the bbox by only `tol` pixels,
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

    def get_image_header(self) -> Optional[pymupdf.Rect]:
        """Find if any image is the header. See #26.

        Basically we go through all svg images on the page, and filter in the ones that are very
        wide and have reasonable height to be a header image. For those images, we keep the ones w/
        grey-ish background. In principle, only the header image can meet these criteria.

        :return: None if found things. Else return the coords. of the image
        """
        images = []
        for img in self.drawings:
            if img['rect'].width > self.bound()[2] * 0.8 \
                    and 10 < img['rect'].height < 50 \
                    and np.isclose(img['fill'], [0.72, 0.72, 0.72], rtol=0.1).all():
                images.append(img)
        assert len(images) <= 1, f'found more than one header image on page {self.number} in ' \
                                 f'{self.parent.name}'
        if images:
            return images[0]['rect']
        else:
            return None


class ParsingError(Exception):
    pass


if __name__ == '__name__':
    pass
