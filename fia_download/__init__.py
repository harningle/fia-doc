import datetime
import hashlib
import os
import re
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Optional

import requests

from .base import get_url, Document
from .decision_docs import load_decision_documents
from .timing_docs import load_timing_documents


class FIADocumentsInterface:
    def __init__(self):
        self.backend = FileSystemDocumentBackend("C:/Dateien/Code/fia-doc/data")

    def update_decision_documents(
            self,
            season: int,
            event: Optional[str] = None,
            session: Optional[str] = None
    ):
        decision_docs = load_decision_documents(season, event)
        updated_docs: dict[str, list] = defaultdict(list)
        for event_name, docs in decision_docs.items():
            for document in docs.values():
                updated = self.backend.update(document, season, event_name)
                if updated:
                    updated_docs[event_name].append(document.name)

        return dict(updated_docs)

    def update_timing_documents(
            self,
            season: int,
            event: str,
            session: str
    ):
        timing_docs = load_timing_documents(season, event, session)
        updated_docs: dict[str, list] = defaultdict(list)
        for session_name, docs in timing_docs.items():
            for document in docs.values():
                updated = self.backend.update(document, season, event, session_name)
                if updated:
                    updated_docs[session_name].append(document.name)

        return dict(updated_docs)

    def get_document(
            self,
            name: str,
            season: int,
            event: str,
            session: Optional[str] = None
    ) -> bytes:
        pass


class BaseDocumentBackend(ABC):
    def download(self, document: Document) -> bytes:
        url = get_url(document.path)
        return requests.get(url).content

    @abstractmethod
    def update(self,
              document: Document,
              season: int,
              event: str,
              session: Optional[str] = None
              ) -> bool:
        pass

    @abstractmethod
    def get(self,
            name: str,
            season: int,
            event: str,
            session: Optional[str] = None
            ) -> bytes:
        pass


class FileSystemDocumentBackend(BaseDocumentBackend):
    FILENAME_REGEX = re.compile(r"(.+?)(?:\((\d{10})\))?\.pdf")
    HEADER_DATE_FORMAT = "%a, %d %b %Y %H:%M:%S GMT"

    def __init__(self, base_dir: str, always_hash: bool = False):
        self.base_dir = base_dir
        self.always_hash = always_hash

    def update(self,
              document: Document,
              season: int,
              event: str,
              session: Optional[str] = None
              ) -> bool:
        # if document has a published date, add it to the filename
        if document.published_at is not None:
            dt_str = datetime.datetime.strftime(
                document.published_at, "%y%m%d%H%M"
            )
            file_name = document.name + f"({dt_str})" + ".pdf"
        else:
            file_name = document.name + ".pdf"

        # create path, session may be none for decision documents
        if session is None:
            path = os.path.join(self.base_dir, str(season), event, file_name)
        else:
            path = os.path.join(self.base_dir, str(season), event, session, file_name)

        os.makedirs(os.path.dirname(path), exist_ok=True)

        # if the document is already downloaded and there is a published date,
        # check if the document is already up-to-date by filename
        if ((document.published_at is not None)
                and os.path.exists(path)
                and not self.always_hash):
            # document does not need to be updated
            return False

        old_path, old_published_at = self._match_document_name(document, path)

        # TODO: maybe work with Last-Modified and If-Modified-Since headers
        content = self.download(document)

        # if there was no published date, we enforce hashing compare by hash
        if old_path and ((document.published_at is None) or self.always_hash):
            with open(path, "rb") as f:
                old_content = f.read()
            new_hash = hashlib.md5(content).hexdigest()
            old_hash = hashlib.md5(old_content).hexdigest()
            if new_hash == old_hash:
                # document does not need to be updated
                return False

        # remove old versions of the document
        if old_path:
            os.remove(old_path)

        # write the new document
        with open(path, "wb") as f:
            f.write(content)

        return True

    def _match_document_name(self, document, path) -> tuple[str, str] | tuple[None, None]:
        for file_name in os.listdir(os.path.dirname(path)):
            # regex match the file list, document name may or may not
            # contain a published date
            match = self.FILENAME_REGEX.match(file_name)
            if match is None:
                continue
            if match.group(1) == document.name:
                file_path = os.path.join(os.path.dirname(path), file_name)
                return file_path, match.group(2)
        return None, None

    def get(self,
            name: str,
            season: int,
            event: str,
            session: Optional[str] = None
            ) -> bytes:
        pass
