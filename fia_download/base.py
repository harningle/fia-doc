import dataclasses
import datetime
import unicodedata
import webbrowser

from dataclasses import dataclass, field
from typing import Optional


BASE_URL = "https://www.fia.com"


@dataclass
class Document:
    """A class representing a document on the FIA website."""
    name: str
    path: Optional[str]
    season: int
    event: str
    session: Optional[str] = None
    last_modified: Optional[datetime.datetime] = field(default=None)
    etag: Optional[str] = field(default=None)
    hash: Optional[str] = field(default=None)
    content: Optional[bytes] = field(default=None)
    was_updated: Optional[bool] = field(default=None)

    @property
    def is_decision_document(self) -> bool:
        """Returns True if the document is a decision document."""
        return self.session is None

    @property
    def is_timing_document(self) -> bool:
        """Returns True if the document is a timing document."""
        return self.session is not None

    @property
    def is_provisional(self) -> bool:
        """Returns True if the document is a provisional document."""
        return "provisional".casefold() in self.name.casefold()

    def open_in_browser(self):
        """Open the document in the default web browser."""
        # ### doesn't work because data URI is too long (on windows?)
        # if self.content:
        #     data = base64.b64encode(self.content).decode()
        #     url = f"data:application/pdf;base64,{data}"
        #     webbrowser.open(url)
        if self.path:
            webbrowser.open(get_url(self.path))

    def to_json(self):
        tmp = dataclasses.asdict(self)
        tmp.pop("content")
        tmp.pop("was_updated")
        tmp["last_modified"] = last_modified_from_dt(self.last_modified)
        return tmp


def normalize_string(st: str) -> str:
    """Removes accents from a string and returns the closest possible
    ascii representation.

    (see also https://stackoverflow.com/a/518232)

    Args:
        st: A string.

    Returns:
        String normalized to the closes possible ascii representation.
    """
    stripped = ''.join(c for c in unicodedata.normalize('NFD', st)
                       if unicodedata.category(c) != 'Mn')
    stripped = stripped.replace("´", "'").replace("`", "'")
    return stripped


def get_url(*paths):
    """Create a FIA server URL from a list of path components.

    Args:
        *paths: one or multiple path components.

    Returns:
        A URL string.
    """
    paths = list(paths)
    for i in range(len(paths)):
        paths[i] = paths[i].lstrip('/').rstrip('/')

    return BASE_URL + '/' + '/'.join(paths)


HEADER_DATE_FORMAT = "%a, %d %b %Y %H:%M:%S GMT"
"""Format string of Last-Modified header values."""


def dt_from_last_modified(last_modified: str) -> datetime.datetime:
    """Create a datetime object from a Last-Modified header value.

    Args:
        last_modified: A Last-Modified header value (string).

    Returns:
        A datetime object.
    """
    return datetime.datetime.strptime(last_modified, HEADER_DATE_FORMAT)


def last_modified_from_dt(dt: datetime.datetime) -> str:
    """Convert a datetime object to a Last-Modified header value.

    Args:
        dt: A datetime object.

    Returns:
        A Last-Modified header value (string).
    """
    return dt.strftime(HEADER_DATE_FORMAT)
