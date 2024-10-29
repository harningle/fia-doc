import requests

import bs4

# TODO: switch to absolute imports once this is a package
from .base import BASE_URL, Document, get_url


def get_calendar_path(year: int):
    """Returns the path to the calendar page for a given year."""
    return get_url(f"/events/fia-formula-one-world-championship/"
                   f"season-{year}/formula-one")


# mapping of normalized section names in the HTML to the session names used in
# this module
_SESSIONS = {
    "RACE": "race",
    "QUALIFYING": "qualifying",
    "SPRINTRACE": "sprint race",
    "SPRINTQUALIFYING": "sprint qualifying",
    "THIRDPRACTICE": "practice 3",
    "SECONDPRACTICE": "practice 2",
    "FIRSTPRACTICE": "practice 1",
}


def load_timing_documents(
        season: int,
        event: str,
        session: str
) -> dict[str, Document]:
    """Loads timing documents metadata for all timing documents of a given
    season, event and session by scraping the FIA website.

    Args:
        season: The season year.
        event: The event name.
        session: The session name.

    Returns:
        A dictionary of Document objects by their document name.
    """
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

    current_session = None
    session_docs: dict[str, Document] = {}

    results = (soup
               .find(class_='content')
               .find(class_='middle')
               .find_all(['p', 'div'], recursive=False))

    for tag in results:
        txt = (tag.text.replace("SESSION", "")
                       .replace(" ", "")
                       .replace(" ", "")
                       .upper())

        if (tag.name == 'p') and (txt in _SESSIONS):
            current_session = _SESSIONS[txt]

        elif (current_session == session) and (tag.name == 'div'):
            url = tag.find('a')['href'].replace(BASE_URL, "")
            title = tag.find('a').find(class_='title').text
            session_docs[title] = Document(title, url, season, event, session)

    return session_docs
