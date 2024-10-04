import datetime
import re
import zoneinfo

from typing import Optional

import requests

import bs4

from .base import BASE_URL, Document, normalize_string, get_url


def _get_season_ids() -> dict[int, int]:
    url = get_url('documents')  # /documents redirects to current season
    response = requests.get(url)
    soup = bs4.BeautifulSoup(response.text, "html.parser")

    season_ids: dict[int, int] = {}

    html_year_select = soup.find_all(id='facetapi_select_facet_form_3')[0]
    for option in html_year_select.find_all('option'):
        path = option.get('value')
        if path == "0":
            continue  # skip placeholder
        if "/season/" not in path:
            # special case, current season has no full path in selector, use
            # the current URL
            path = response.url.replace(BASE_URL, "")
        season_id = int(path[-4:])
        year = int(option.text.lstrip('SEASON '))
        season_ids[year] = season_id

    return season_ids


def load_decision_documents(season: int, event: Optional[str] = None):
    docs: dict[str, dict[str, Document]] = {}

    season_id = _get_season_ids()[season]

    path = (f"/documents/championships/fia-formula-one-world-championship-14/"
            f"season/season-{season}-{season_id}")
    url = get_url(path)
    response = requests.get(url)
    soup = bs4.BeautifulSoup(response.text, "html.parser")

    html_events_list = soup.find(class_='decision-document-list')
    for html_event in html_events_list.find_all('ul'):
        try:
            event_name = html_event.find(class_='event-title').text
        except AttributeError:
            continue  # skip non-event items

        docs[event_name] = {}

        if (event is not None) and (event_name != event):
            continue

        event_docs: dict[str, Document] = {}

        for doc_html in html_event.find_all(class_="document-row"):
            document = html_parse_document(doc_html)

            if document is None:
                continue

            if ((document.name in event_docs)
                    and (event_docs[document.name].published_at
                         >= document.published_at)):
                # a newer version of the document already exists
                continue

            event_docs[document.name] = document

        docs[event_name] = event_docs

    if event not in docs:
        raise ValueError(f"Event '{event}' not found in season {season}")

    return docs


doc_title_regex = re.compile(
    r"(?:Doc (\d+) - )?"        # optional document number, e.g. "Doc 12 - "
    r"(.+?(?= [vV]\d|$))"       # document title excluding optional version  
    r"(?: [vV](\d))?"           # optional version number, e.g. " v2"
)


def html_parse_document(doc_html) -> Document | None:
    title = re.findall(r'\n*(\w.+)', doc_html.find(class_="title").text)[0]
    # file_type = doc_html.find(class_="file-type").div['class'][0]
    published = doc_html.find(class_="published").span.text
    try:
        path = doc_html.a.get('href')
    except AttributeError:
        path = None

    title = normalize_string(title)

    if title.startswith("Recalled - "):
        return None

    number, title, version = re.search(doc_title_regex, title).groups()
    # number is only available for newer docs and not used
    # version is not used, instead we rely on publish date

    published_at = datetime.datetime.strptime(
        published, "%d.%m.%y %H:%M"
    ).replace(tzinfo=zoneinfo.ZoneInfo("CET"))

    return Document(title, path, published_at)
