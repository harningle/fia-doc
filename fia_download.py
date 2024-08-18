import datetime
import re
from dataclasses import dataclass, field
from typing import Optional

import requests

import bs4


BASE_URL = "https://www.fia.com"


@dataclass
class Season:
    year: int
    path: str
    _events: dict[str, "Event"] = field(default_factory=dict, repr=False)

    def add_event(self, name: str):
        event = Event(name=name, season=self)
        self._events[name] = event
        return event

    def get_event(self, name: str):
        if name not in self._events:
            return None
        return self._events[name]


@dataclass
class Event:
    # is_testing: bool
    # round_: int | None
    name: str
    # path: str
    season: "Season"
    _documents: dict[str, "Document"] = field(default_factory=dict, repr=False)

    def add_document(self, name: str) -> "Document":
        document = Document(name=name)
        self._documents[name] = document
        return document

    def get_document(self, name: str) -> Optional["Document"]:
        if name not in self._documents:
            return None
        return self._documents[name]

@dataclass
class Document:
    name: str
    _versions: list[tuple[datetime.datetime, "DocumentVersion"]] \
        = field(default_factory=list, repr=False)

    def get_current_version(self) -> Optional["DocumentVersion"]:
        if not self._versions:
            return None
        return self._versions[-1][1]

    def get_versions(self) -> list["DocumentVersion"]:
        return [v for _, v in self._versions]

    def add_version(self,
                    published_at: datetime.datetime,
                    is_recalled: bool,
                    number: int | None,
                    path: str | None) -> "DocumentVersion":

        latest_version = self.get_current_version()
        if latest_version and (latest_version.published_at >= published_at):
            is_current = True
            # mark other versions as not current
            for _, v in self._versions:
                v.is_current = False
        else:
            is_current = False

        version = DocumentVersion(document=self,
                                  published_at=published_at,
                                  is_recalled=is_recalled,
                                  is_current=is_current,
                                  number=number,
                                  path=path)

        self._versions.append((version.published_at, version))
        self._versions.sort(key=lambda x: x[0])

        return version


@dataclass
class DocumentVersion:
    document: "Document"
    published_at: datetime.datetime
    is_recalled: bool
    is_current: bool
    number: int | None
    path: str | None


def get_url(*paths):
    paths = list(paths)
    for i in range(len(paths)):
        paths[i] = paths[i].lstrip('/').rstrip('/')

    return BASE_URL + '/' + '/'.join(paths)


def list_seasons():
    url = get_url('documents')  # /documents redirects to current season
    response = requests.get(url)
    soup = bs4.BeautifulSoup(response.text, "html.parser")

    seasons = []

    html_year_select = soup.find_all(id='facetapi_select_facet_form_3')[0]
    for option in html_year_select.find_all('option'):
        path = option.get('value')
        if path == "0":
            continue  # skip placeholder
        if "/season/" not in path:
            # special case, current season has no full path in selector, use
            # the current URL
            path = response.url.replace(BASE_URL, "")
        year = int(option.text.lstrip('SEASON '))
        seasons.append(Season(year=year, path=path))

    return seasons


def load_documents(year: int):
    seasons = list_seasons()

    for season in seasons:
        if season.year == year:
            break
    else:
        raise ValueError(f"Season {year} not found")

    url = get_url(season.path)
    response = requests.get(url)
    soup = bs4.BeautifulSoup(response.text, "html.parser")

    events = []

    html_events_list = soup.find(class_='decision-document-list')
    for html_event in html_events_list.find_all('ul'):
        try:
            event_name = html_event.find(class_='event-title').text
        except AttributeError:
            continue  # skip non-event items

        event = season.add_event(name=event_name)

        for doc_html in html_event.find_all(class_="document-row"):
            html_parse_document(doc_html, event)

        events.append(event_name)

    return season


def html_parse_document(doc_html, event: "Event"):
    title = re.findall(r'\n*(\w.+)', doc_html.find(class_="title").text)[0]
    # file_type = doc_html.find(class_="file-type").div['class'][0]
    published = doc_html.find(class_="published").span.text
    try:
        path = doc_html.a.get('href')
    except AttributeError:
        path = None

    # TODO: convert title to ascii before further processing

    status, number, title = re.search(
        r'(?:(?!Doc)([\w ]+) - )?(?:Doc (\d+) - )?([\w ]+)', title
    ).groups()

    status = "" if status is None else status
    number = None if number is None else int(number)

    published_at = datetime.datetime.strptime(
        published, "%d.%m.%y %H:%M"
    ).replace(tzinfo=datetime.UTC)

    is_recalled = "Recalled" in status

    document = event.get_document(title)
    if document is None:
        document = event.add_document(name=title)

    document.add_version(published_at=published_at,
                         is_recalled=is_recalled,
                         number=number,
                         path=path)
