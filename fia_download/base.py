import datetime
import unicodedata
import webbrowser
from dataclasses import dataclass, field

import requests


BASE_URL = "https://www.fia.com"


@dataclass
class Document:
    name: str
    path: str
    published_at: datetime.datetime | None


def normalize_string(name: str) -> str:
    # removes accents from a string and returns the closest possible
    # ascii representation (https://stackoverflow.com/a/518232)
    stripped = ''.join(c for c in unicodedata.normalize('NFD', name)
                       if unicodedata.category(c) != 'Mn')
    stripped = stripped.replace("´", "'").replace("`", "'")
    return stripped


def get_url(*paths):
    paths = list(paths)
    for i in range(len(paths)):
        paths[i] = paths[i].lstrip('/').rstrip('/')

    return BASE_URL + '/' + '/'.join(paths)
