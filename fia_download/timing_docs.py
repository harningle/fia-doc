import requests

import bs4

from .base import BASE_URL, Document, get_url


def get_calendar_path(year: int):
    return get_url(f"/events/fia-formula-one-world-championship/"
                   f"season-{year}/formula-one")


_SESSIONS = {
    "RACE": "race",
    "QUALIFYING": "qualifying",
    "SPRINTRACE": "sprint race",
    "SPRINTQUALIFYING": "sprint qualifying",
    "THIRDPRACTICE": "practice 3",
    "SECONDPRACTICE": "practice 2",
    "FIRSTPRACTICE": "practice 1",
}


def load_timing_documents(season: int, event: str, session: str):
    if season < 2022:
        raise ValueError("Timing documents are only supported from 2022 "
                         "onwards")

    event_slug = event.lower().replace(" ", "-")

    url = get_url(
        f"events/fia-formula-one-world-championship/"
        f"season-{season}/{event_slug}/eventtiming-information"
    )
    response = requests.get(url)
    soup = bs4.BeautifulSoup(response.text, "html.parser")

    session = None
    event_docs: dict[str, dict[str, Document]] = {}

    results = (soup
               .find(class_='content')
               .find(class_='middle')
               .find_all(['p', 'div'], recursive=False))

    for tag in results:
        txt = tag.text.replace("SESSION", "").replace(" ", "").replace(" ", "").upper()
        if (tag.name == 'p') and (txt in _SESSIONS):
            session = _SESSIONS[txt]
            event_docs[session] = {}
        elif session and (tag.name == 'div'):
            url = tag.find('a')['href'].replace(BASE_URL, "")
            title = tag.find('a').find(class_='title').text
            event_docs[session][title] = Document(title, url, None)

    return event_docs
