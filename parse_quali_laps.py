# -*- coding: utf-8 -*-
import os

import fitz
import pandas as pd


def parse_quali_final_classification(file: str | os.PathLike) -> pd.DataFrame:
    """Parse "Qualifying Session Final Classification" PDF"""
    # Find the page with "Qualifying Session Final Classification"
    doc = fitz.open(file)
    found = None
    for i in range(len(doc)):
        page = doc[i]
        found = page.search_for('Qualifying Session Final Classification')
        if found:
            break
    if found is None:
        raise ValueError(f'not able to find quali. result in `{file}`')

    # Width and height of the page
    w, h = page.bound()[2], page.bound()[3]

    # y-position of "Qualifying Final Classification"
    y = found[0].y1

    # y-position of "NOT CLASSIFIED - " or "POLE POSITION LAP"
    not_classified = page.search_for('NOT CLASSIFIED - ')
    b = None
    if not_classified:
        b = not_classified[0].y0
    else:
        b = page.search_for('POLE POSITION LAP')[0].y0
    if b is None:
        raise ValueError(f'not able to find the bottom of quali. result in `{file}`')

    # Table bounding box
    bbox = fitz.Rect(0, y, w, b)

    # Dist. between "NAT" and "ENTRANT"
    nat = page.search_for('NAT')[0]
    entrant = page.search_for('ENTRANT')[0]
    snap_x_tolerance = (entrant.x0 - nat.x1) * 1.2  # 20% buffer

    # Parse
    df = page.find_tables(clip=bbox, snap_x_tolerance=snap_x_tolerance)[0].to_pandas()
    df.columns = ['_', 'NO', 'DRIVER', 'NAT', 'ENTRANT', 'Q1', 'Q1_LAPS', 'Q1_%', 'Q1_TIME', 'Q2',
                  'Q2_LAPS', 'Q2_TIME', 'Q3', 'Q3_LAPS', 'Q3_TIME']
    df.drop(columns=['_', 'NAT', 'ENTRANT'], inplace=True)
    df = df[df['NO'] != '']
    return df


def get_strikeout_text(page: fitz.Page) -> list[tuple[fitz.Rect, str]]:
    """Get the strikeout text and their locations in the page

    See https://stackoverflow.com/a/74582342/12867291.

    :param page: fitz.Page object
    :return: A list of tuples, where each tuple is the bbox and the strikeout text
    """
    # Get all strikeout lines
    lines = []
    paths = page.get_drawings()  # Strikeout lines are in fact vector graphics. To be more precise,
    for path in paths:           # they are rectangles with very small height
        for item in path['items']:
            if item[0] == 're':  # If a graphic is a rectangle, check if a horizontal line
                rect = item[1]
                if rect.width <= 2 * rect.height or rect.height > 1:
                    continue     # If width-to-height ratio is less than 2, it's not a line
                lines.append(rect)

    # Get all texts on this page
    # TODO: the O(n^2) here can probably be optimised later
    words = page.get_text('words')
    strikeout = []
    for rect in lines:
        for w in words:
            text_rect = fitz.Rect(w[:4])  # bbox of the word
            text = w[4]  # word text
            if text_rect.intersects(rect):
                strikeout.append((rect, text))
    return strikeout


def parse_quali_lap_times_page(page: fitz.Page) -> pd.DataFrame:
    """Parse a page in "Qualifying Session Lap Times" PDF

    This is the most complicated parsing. See `notebook/demo.ipynb` for more details.
    """
    # Page width. All tables are horizontally bounded in [0, `w`]
    w = page.bound()[2]

    # Positions of "NO" and "TIME". They are the top of each table. One driver may have multiple
    # tables starting from roughly the same top position
    no_time_pos = page.search_for('NO TIME')
    ys = [i.y1 for i in no_time_pos]
    ys.sort()                      # Sort these "NO"'s from top to bottom
    top_pos = [ys[0] - 1]          # -1 to give a little buffer at the top
    for y in ys[1:]:
        if y - top_pos[-1] > 10:   # Many "No"'s are at the same height, and we only need those at
            top_pos.append(y - 1)  # different heights. If there is 10px gap, we take it as diff.

    # Bottom of the table is the next "NO TIME", or the bottom of the page
    ys = [i.y0 for i in no_time_pos]
    ys.sort()
    bottom_pos = [ys[0]]
    for y in ys[1:]:
        if y - bottom_pos[-1] > 10:
            bottom_pos.append(y)
    b = page.bound()[3]
    bottom_pos.append(b)
    bottom_pos = bottom_pos[1:]  # The first "NO TIME" is not the bottom of any table

    # Find the tables located between `top_pos` and `bottom_pos`
    df = []
    for row in range(len(top_pos)):
        for col in range(3):  # Three drivers in one row
            # Find the driver name, which is located immediately above the table
            driver = page.get_text(
                'block',
                clip=fitz.Rect(
                    col * w / 3,
                    top_pos[row] - 30,
                    (col + 1) * w / 3,
                    top_pos[row] - 10
                )
            ).strip()
            if not driver:  # In the very last row may not have all three drivers
                continue
            car_no, driver = driver.split(maxsplit=1)

            # Find tables in the bounding box of driver i's table
            tabs = page.find_tables(
                clip=fitz.Rect(col * w / 3, top_pos[row], (col + 1) * w / 3, bottom_pos[row]),
                strategy='lines'
            )  # TODO: I think this may fail if a driver only has one lap and crashes...

            # Check if any lap times is strikeout
            # TODO: the O(n^2) here can probably be optimised later
            # TODO: we should introduce the colour check as well: the strikeout texts are in grey,
            #       while normal texts are black
            strikeout_texts = get_strikeout_text(page)
            temp = []
            lap_time_deleted = []
            for tab in tabs:
                for r in tab.rows:
                    assert len(r.cells) == 3  # Should have lap, if pit, and lap time columns
                    cell = r.cells[2]  # Lap time cell (the last) can be strikeout
                    is_strikeout = False
                    bbox = fitz.Rect(cell)
                    for rect, text in strikeout_texts:
                        if bbox.intersects(rect):
                            is_strikeout = True
                            break
                    lap_time_deleted.append(is_strikeout)
                temp.append(tab.to_pandas())

            # One driver may have multiple tables. Concatenate them
            temp = pd.concat(temp, ignore_index=True)
            temp.columns = ['lap_no', 'pit', 'lap_time']
            temp['car_no'] = car_no
            temp['driver'] = driver
            temp['lap_time_deleted'] = lap_time_deleted
            df.append(temp)
    return pd.concat(df, ignore_index=True)


def parse_quali_lap_times(file: str | os.PathLike) -> pd.DataFrame:
    """Parse "Qualifying Session Lap Times" PDF"""
    # Get page width and height
    doc = fitz.open(file)

    # Parse all pages
    tables = []
    for page in doc:
        tables.append(parse_quali_lap_times_page(page))
    df = pd.concat(tables, ignore_index=True)
    return df


if __name__ == '__main__':
    pass
