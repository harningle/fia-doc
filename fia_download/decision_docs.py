import datetime
import functools
import re
import zoneinfo

from typing import Optional

import requests

import bs4

# TODO: switch to absolute imports once this is a package
from .base import BASE_URL, Document, normalize_string, get_url


def get_season_ids() -> dict[int, int]:
    """Get a mapping of season years to season IDs.

    The FIA website uses non-intuitive season IDs in their URLs to identify
    seasons. This function scrapes the season selector on the documents page
    to get the correct season IDs for each season year.

    Returns:
        A dictionary mapping season years to season IDs.
    """
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


@functools.cache
def load_decision_documents(
        season: int,
        event: Optional[str] = None
) -> dict[str, dict[str, Document]]:
    """Loads decision documents metadata for all decision documents of a given
    season and event by scraping the FIA website.

    Args:
        season: The season year.
        event: The event name. If None, all events are loaded.

    Returns:
        A dictionary that maps event names to dictionaries that map document
        names to Document objects.
        E.g. ``{"Event Name": {"Document Name": <obj>, ...}, ...}``
    """
    docs: dict[str, dict[str, Document]] = {}

    season_id = get_season_ids()[season]

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

        # skip events that don't match the requested event name (if specified)
        if (event is not None) and (event_name != event):
            continue

        event_docs: dict[str, Document] = {}

        for doc_html in html_event.find_all(class_="document-row"):
            if (docinfo := html_parse_document(doc_html)) is not None:
                title, path, published_at = docinfo
                # Create a Document object with a preliminary last_modified
                # value from the data given in the HTML. This is used here to
                # filter out older versions of the same document. The value
                # is later updated with the more precise last modified date
                # from the HTTP response headers.
                document = Document(title, path, season, event_name,
                                    last_modified=published_at)
            else:
                continue

            if ((document.name in event_docs)
                    and (event_docs[document.name].last_modified
                         >= document.last_modified)):
                # a newer version of the document already exists
                continue

            event_docs[document.name.casefold()] = document

        docs[event_name] = event_docs

    # if a specific event was requested, check if it was found
    if (event is not None) and (event not in docs):
        raise ValueError(f"Event '{event}' not found in season {season}")

    return docs


doc_title_regex = re.compile(
    r"(?:Doc (\d+) - )?"        # optional document number, e.g. "Doc 12 - "
    r"(.+?(?= [vV]\d|$))"       # document title excluding optional version  
    r"(?: [vV](\d))?"           # optional version number, e.g. " v2"
)


def html_parse_document(doc_html) -> tuple[str, str, datetime.datetime] | None:
    """Parse a document HTML element and return the document metadata.

    Args:
        doc_html: A BeautifulSoup element representing a document entry on the
            FIA webpage.

    Returns:
        A tuple containing the document title, path and publish date.
        If the document is a recalled document or has no PDF download URL,
        None is returned instead.
    """
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

    return title, path, published_at
