import os
import requests
import pandas as pd
import zipfile
from parse_race_history_chart import parse_race_history_chart

data_dir = "./data"


def download_historical_data() -> None:
    if not os.path.exists(f"{data_dir}/lap_times.csv"):
        resp = requests.get("https://ergast.com/downloads/f1db_csv.zip", stream=True)
        with open("f1db_csv.zip", "wb") as f:
            for chunk in resp.iter_content(chunk_size=4096):
                f.write(chunk)
        os.makedirs(data_dir, exist_ok=True)
        with zipfile.ZipFile("f1db_csv.zip", "r") as zip_ref:
            zip_ref.extractall(data_dir)
        os.remove("f1db_csv.zip")


def historical_lap_times(round: int) -> pd.DataFrame:
    # 2023 lap times
    df_lap_times = pd.read_csv(
        f"{data_dir}/lap_times.csv", usecols=["raceId", "driverId", "lap", "time"]
    )
    df_races = pd.read_csv(f"{data_dir}/races.csv", usecols=["raceId", "year", "round"])
    df_races = df_races[df_races["year"] == 2023]
    df_lap_times = df_lap_times.merge(df_races, on="raceId", how="inner")
    del df_races
    df_drivers = pd.read_csv(f"{data_dir}/drivers.csv", usecols=["driverId", "number"])
    df_lap_times = df_lap_times.merge(df_drivers, on="driverId", how="inner")
    df_lap_times = df_lap_times.drop(columns=["raceId", "driverId", "year"])
    # TODO: Parse more than one race history chart
    df_lap_times = df_lap_times.loc[df_lap_times["round"] == round, :]
    df_lap_times.loc[:, "number"] = df_lap_times["number"].astype(int)
    df_lap_times.loc[df_lap_times["number"] == 33, "number"] = (
        1  # Ver: 33 --> 1 in Ergast
    )
    return df_lap_times


def test_historical_lap_times():
    download_historical_data()
    df_historical_lap_times = historical_lap_times(22)
    df_parsed_lap_times = parse_race_history_chart("fia_pdfs/race_history_chart.pdf")
    df_matched = df_historical_lap_times.merge(
        df_parsed_lap_times,
        left_on=["lap", "number"],
        right_on=["lap", "driver_no"],
        how="outer",
        indicator=True,
    )
    assert len(df_matched[df_matched["_merge"] != "both"]) == 0
