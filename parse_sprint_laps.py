# -*- coding: utf-8 -*-
import os
import pickle

import fitz
import pandas as pd

from models.foreign_key import SessionEntry
from models.quali_lap import Lap, LapData


def parse_sprint_lap_analysis_page(page: fitz.Page) -> pd.DataFrame:
    """Parse a page in "Sprint Lap Analysis" PDF"""
    w = page.bound()[2]                               # Page width. Shared by three drivers
    t = page.search_for('Sprint Lap Analysis')[0].y1  # "Race Lap Analysis" at the top of the page
    b = page.search_for('Page')[0].y0                 # "Page" at the bottom right of the page

    # Find the tables inside the left/middle/right part of the page
    """
    On the page, the drivers and tables are laid out in the following way:
    
    | driver 1              | driver 2              | driver 3              |
    | table 1.1 | table 1.2 | table 2.1 | table 2.2 | table 3.1 | table 3.2 |
    | driver 4              | driver 5              | driver 6              |
    | table 4.1 | table 4.2 | table 5.1 | table 5.2 | table 6.1 | table 6.2 |
    | ...
    
    Horizontally, there are three parts: left, middle, and right. Each part contains a driver's
    name and his tables (or perhaps only one if the race only has very few laps). Sometimes (if we
    have a lot of laps so the tables are tall), one part only has one driver vertically. In other
    times (only a few laps, as in the sprint, so the tables are short), one part can have multiple
    drivers vertically.
    
    TODO: we assume pymupdf returns tables in the order of 1.1, 1.2, 2.1, 2.2 below. Should check
    if this is always the case.
    """
    df = []
    for i in range(3):
        # Find all tables in this part
        l = i * w / 3
        r = (i + 1) * w / 3
        tabs = page.find_tables(clip=fitz.Rect(l, t, r, b), strategy='lines')
        if not tabs.tables:  # E.g., 20 drivers, 3 parts, 20 % 3 != 0, so will not all parts have
            continue         # driver. TODO: this can be tested, e.g., in this example we should
                             # only has one part with no driver
        # Find the driver's name and link tables to the driver
        assert tabs[0].to_pandas().iloc[0, 0] == '1'  # The 1st table's 1st cell should be lap #. 1
        driver_tabs = []  # Store the tables for the driver being processed currently
        for tab in tabs:
            if tab.to_pandas().iloc[0, 0] == '1':   # The 1st table for a driver has lap #. 1
                # If we already have some tables and now find a new driver, then the tables belong
                # to the previous driver
                if driver_tabs:
                    for j in driver_tabs:
                        j['driver'] = name
                        j['car_no'] = car_no
                        del j['Col1']
                        df.append(j)
                    driver_tabs = []
                h = tab.header.bbox[1]  # Top of the table
                name = page.get_text('block', clip=fitz.Rect(l, h - 30, r, h)).strip()
                car_no, name = name.split('\n')
            driver_tabs.append(tab.to_pandas())

        # Process the last driver in this part. His tables won't be appended in the loop above
        for j in driver_tabs:
            j['driver'] = name
            j['car_no'] = car_no
            df.append(j)

    return pd.concat(df, ignore_index=True)


def parse_sprint_lap_analysis(file: str | os.PathLike) -> pd.DataFrame:
    """Parse "Sprint Lap Analysis" PDF"""
    doc = fitz.open(file)
    df = []
    for page in doc:
        df.append(parse_sprint_lap_analysis_page(page))
    return pd.concat(df, ignore_index=True)
