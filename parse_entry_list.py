# -*- coding: utf-8 -*-
import os
import pickle

import fitz
import pandas as pd

from models.driver import Driver, RoundEntry
from models.foreign_key import Round


def split_team_constructor(row_data: list[str]) -> list[str]:
    """
    Split cases where pymupdf combines the row, constructor columns
    TODO: There has to be a more elegant way to fix this
    """
    team_constructor = row_data[3]

    possible_team_endings = ["Racing", "Team"]
    for team_ending in possible_team_endings:
        if team_ending in team_constructor:
            split_team_constructor = team_constructor.split(team_ending)
            team = split_team_constructor[0] + team_ending
            constructor = split_team_constructor[1].strip()
            new_row_data = row_data[0:3] + [team, constructor]
            return new_row_data
    raise ValueError(f"Unable to split team and constructor for '{team_constructor}'")

def extract_table_from_bbox(page, bbox) -> list[list[str, int]]:
    blocks = page.get_text("dict", clip=bbox)["blocks"][-1]
    rows = {}
    last_y0 = None
    tolerance = 5
    
    for line in blocks["lines"]:
        for span in line["spans"]:
            x0, y0, x1, y1 = span["bbox"]
            if last_y0 is not None and abs(y0 - last_y0) <= tolerance:
                y0 = last_y0
            else:
                tolerance = max(5, (y1 - y0) / 2)
            row_key = round(y0, -1)
            last_y0 = y0
            if row_key not in rows:
                rows[row_key] = []
            rows[row_key].append((x0, span["text"].strip()))

    sorted_rows = sorted(rows.items(), key=lambda item: item[0])
    table = []
    for _, row_blocks in sorted_rows:
        sorted_row = sorted(row_blocks, key=lambda b: b[0])
        row_data = [text for _, text in sorted_row if text]
        if len(row_data) == 6:
            row_data = row_data[0:1] + row_data[2:]
        elif len(row_data) == 4:
            row_data = split_team_constructor(row_data)
        elif len(row_data) < 4:
            continue

        if row_data[0] != "No.":
            row_data[0] = int(row_data[0])
            
        table.append(row_data)

    return table



def parse_entry_list(file: str | os.PathLike) -> pd.DataFrame:
    """Parse the table from "Entry List" PDF.

    An example of `team` and `constructor` is "Alfa Romeo F1 Team Stake" and "Alfa Romeo Ferrari".

    `role` can be "permanent" or "reserve".

    See `notebook/demo.ipynb` for the detailed explanation of the table structure.

    :param file: Path to PDF file
    :return: A dataframe of [car No., driver name, nationality, team, constructor, role]
    """
    doc = fitz.open(file)
    page = doc[1]  # TODO: can have multiple pages
    w, h = page.bound()[2], page.bound()[3]
    car_no = page.search_for("No.")[0]
    top_left = (car_no.x0, car_no.y0)

    text_height = (car_no.y1 - car_no.y0) * 1.035
    top = car_no.y0 + text_height
    bottom = car_no.y1 + text_height
    no_left = car_no.x0
    no_right = car_no.x1
    while top < h:
        text = page.get_text("text", clip=(no_left, top, no_right, bottom))
        if text.strip():
            top += text_height
            bottom += text_height
        else:
            break
    bottom_right = (w, top)

    # Invert y-coordinates for pymupdf (fitz)
    top_left = (int(top_left[0]), int(h - top_left[1]))
    bottom_right = (int(bottom_right[0]), int(h - bottom_right[1]) + 1)

    # Parse using `pymupdf`
    bbox = (top_left[0], h - top_left[1], bottom_right[0], h - bottom_right[1])
    table_data = extract_table_from_bbox(page, bbox)

    processed_data = [row for row in table_data if len(row) == 5]
    df = pd.DataFrame(processed_data[1:], columns=processed_data[0])

    # Extract reserve drivers
    top += text_height
    bottom += text_height
    while top < h:
        text = page.get_text("text", clip=(no_left, top, no_right, bottom))
        if text.strip():
            top += text_height
            bottom += text_height
        else:
            break
    top_left = (no_left, bottom)
    top += text_height
    bottom = h
    while top < h:
        text = page.get_text("text", clip=(no_left, top, no_right, bottom))
        if text.strip():
            top += text_height
            bottom += text_height
        else:
            break
    bottom_right = (w, top)
    top_left = (int(top_left[0]), int(h - top_left[1]))
    bottom_right = (int(bottom_right[0]), int(h - bottom_right[1]) + 1)
    bbox = (top_left[0], h - top_left[1], bottom_right[0], h - bottom_right[1])
    reserve_table_data = extract_table_from_bbox(page, bbox)

    reserve_processed_data = [row for row in reserve_table_data if len(row) == 5]
    if reserve_processed_data:
        reserve_df = pd.DataFrame(reserve_processed_data, columns=processed_data[0])
        reserve_df["role"] = "reserve"
    else:
        reserve_df = pd.DataFrame(columns=df.columns)

    print(reserve_df)

    df["role"] = "permanent"
    combined_df = pd.concat([df, reserve_df], ignore_index=True)

    doc.close()
    return combined_df


def to_json(df: pd.DataFrame):
    # Hard code 2023 Abu Dhabi for now
    year = 2023
    round_no = 22

    # To json
    df["driver"] = df.apply(
        lambda x: Driver(
            car_number=x["No."], name=x["Driver"], team=x["Team"], role=x["role"]
        ),
        axis=1,
    )
    drivers = df["driver"].tolist()
    round_entry = RoundEntry(
        foreign_keys=Round(year=year, round=round_no), objects=drivers
    ).model_dump()

    with open("entry_list.pkl", "wb") as f:
        pickle.dump(round_entry, f)

    return round_entry


if __name__ == "__main__":
    df = parse_entry_list("fia_pdfs/race_entry_list.pdf")
    df = parse_entry_list("fia_pdfs/race_entry_list_2024.pdf")
    pass
