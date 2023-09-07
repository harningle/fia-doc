from io import BytesIO
import json
import re

import fastf1
from pypdf import PdfReader
import requests


def get_event_note(year: int, race: str, **kwargs) -> list:
    """Find the event note doc. for a given GP

    :param year: int: Year
    :param race: str: Grand Prix full name, e.g. British Grand Prix
    :param kwargs: Optional arguments for `requests.get()`
    :return: list: List of URLs for all potential race directors' event note PDF
    """

    url = f'https://www.fia.com/documents/championships/fia-formula-one-world-championship-14/' \
          f'season/season-{URLS[year]}/event/{race}'
    resp = requests.get(url, **kwargs)

    # Find docs. with "event notes" or "Pirelli" in the title
    docs = re.findall(r'href="(.+?).pdf"', resp.text)
    docs = [doc for doc in docs
            if re.match(r'.*?((event-?_? ?notes)|(pirelli)).*?', doc, re.IGNORECASE)]
    return docs


def get_pdf(url: str, **kwargs) -> bytes:
    """Download a PDF file from the given URL.

    :param url: URL of the PDF FIA document
    :param kwargs: Optional arguments for `requests.get()`
    :return: PDF file content
    """

    # Get the PDF file from the URL
    resp = requests.get(url, **kwargs)

    # Try three times if the above request fails
    cnt = 0
    while not resp.ok and cnt < 3:
        resp = requests.get(url, **kwargs)
        cnt += 1
    return resp.content


def parse_event_pdf(pdf: bytes) -> list[str] | None:
    """Parse the PDF file and see if we can find the tyre compound in it

    :param pdf: PDF file content
    :return: list: compound information, e.g. ["C1", "C2", "C3"], or None if not found
    """

    # Go through the pages and look for tyre compound
    pdf = BytesIO(pdf)
    reader = PdfReader(pdf)
    for page in reader.pages:
        text = page.extract_text()
        if 'Compound' in text:
            # See # See https://stackoverflow.com/a/11430936/12867291
            compound = set(re.findall(r'(?=\D(C\d)\D)', text))
            return list(compound)


def get_compound(year: int, race: str, **kwargs) -> list[str] | None:
    """Parse the PDF file and see if we can find the tyre compound in it

    :param year: Year
    :param race: GP name, e.g. British
    :param kwargs: Optional arguments for `requests.get()`
    :return: list: Tyre compounds for this GP, e.g. ["C1", "C2", "C3"], or None if not found
    """

    # Go to the FIA website and find all PDF links
    docs = get_event_note(year, race, **kwargs)

    # Find compound in each PDF file
    for doc in docs:
        url = f'https://www.fia.com{doc}.pdf'
        pdf = get_pdf(url, **kwargs)
        compound = parse_event_pdf(pdf)
        if compound:
            return compound


# URL of the FIA docs. by year
# E.g., for year 2019, the URL is https://www.fia.com/documents/championships/fia-formula-one-
# world-championship-14/season/season-2019-971
URLS = {
    2019: '2019-971',
    2020: '2020-1059',
    2021: '2021-1108',
    2022: '2022-2005',
    2023: '2023-2042'
}


if __name__ == '__main__':
    # Get tyre compounds for all races since 2019
    tyres = {}
    for each_year in range(2019, 2024):
        tyres[each_year] = {}
        races = fastf1.get_event_schedule(each_year)
        for each_race in races.EventName:

            # Handle special cases
            match each_race:
                case '70th Anniversary Grand Prix':
                    normalised_race_name = 'Formula 1 70th Anniversary Grand Prix'
                case 'Mexico City Grand Prix':
                    normalised_race_name = 'Mexican Grand Prix'
                case 'São Paulo Grand Prix':
                    normalised_race_name = 'Brazilian Grand Prix'
                case 'Saudi Arabian Grand Prix':
                    normalised_race_name = 'Saudi Arabia Grand Prix'
                case _:
                    normalised_race_name = each_race
            if each_year >= 2022 and normalised_race_name == 'Saudi Arabia Grand Prix':
                normalised_race_name = 'Saudi Arabian Grand Prix'

            # Skip winter testing
            if 'pre-season' not in normalised_race_name.lower():
                tyres[each_year][each_race] = get_compound(each_year, normalised_race_name)

    # Save to disk
    with open('tyres.json', 'w') as f:
        json.dump(tyres, f, indent=4)
