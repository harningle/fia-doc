import hashlib
import json
import os
import re
import warnings
from typing import Optional

import requests

from rapidfuzz import fuzz

# TODO: switch to absolute imports once this is a package
from .base import (
    Document,
    dt_from_last_modified,
    get_url,
    last_modified_from_dt
)
from .decision_docs import load_decision_documents
from .timing_docs import load_timing_documents


class FIADocumentsInterface:
    """Interface for loading FIA documents.


    """
    def __init__(self, backend_path: str):
        self._backend = FileSystemDocumentBackend(backend_path)

    def check_updated(
            self,
            name: str,
            season: int,
            event: str,
            session: Optional[str] = None,
            allow_fuzzy: bool = False
    ) -> bool:
        """Check if the given document has been updated on the server since
        the last time it was loaded.

        Args:
            name: The name of the document.
            season: The season in which the event took place.
            event: The name of the event.
            session: The session name. If None, a decision document is assumed.
                Else, a timing document is assumed.
            allow_fuzzy: Whether to allow fuzzy matching of the document name.

        Returns:
            True if the document has been updated on the server,
            False otherwise.
        """
        document = self._get_document_metadata(name, season, event, session,
                                               allow_fuzzy=allow_fuzzy)
        return self._backend.was_updated(document)

    def get(
            self,
            name: str,
            season: int,
            event: str,
            session: Optional[str] = None,
            allow_fuzzy: bool = False
    ) -> Document:
        """Load and return the given document.

        Args:
            name: The name of the document.
            season: The season in which the event took place.
            event: The name of the event.
            session: The session name. If None, a decision document is assumed.
                Else, a timing document is assumed.
            allow_fuzzy: Whether to allow fuzzy matching of the document name.

        Returns:
            The loaded document.
        """
        document = self._get_document_metadata(name, season, event, session,
                                               allow_fuzzy=allow_fuzzy)

        if document.is_provisional:
            warnings.warn(f"Document '{name}' is provisional!")

        self._backend.load_document(document)
        return document

    def get_if_updated(
            self,
            name: str,
            season: int,
            event: str,
            session: Optional[str] = None,
            allow_fuzzy: bool = False
    ) -> Document | None:
        """Load and return the given document if it has been updated on the
        server since the last time it was loaded.

        Args:
            name: The name of the document.
            season: The season in which the event took place.
            event: The name of the event.
            session: The session name. If None, a decision
                document is assumed. Else, a timing document is assumed.
            allow_fuzzy: Whether to allow fuzzy matching of the document name.

        Returns:
            The loaded document if it has been updated, None otherwise.
        """
        document = self._get_document_metadata(name, season, event, session,
                                               allow_fuzzy=allow_fuzzy)

        if document.is_provisional:
            warnings.warn(f"Document '{name}' is provisional!")

        if self._backend.was_updated(document):
            self._backend.load_document(document)
            return document
        return None

    @staticmethod
    def _get_document_metadata(
            name: str,
            season: int,
            event: str,
            session: Optional[str] = None,
            allow_fuzzy: bool = False
    ) -> Document:
        name = name.casefold()

        if session is None:
            documents = load_decision_documents(season, event)
            documents = documents.get(event, [])

        else:
            documents = load_timing_documents(season, event, session)

        # exact match
        if name in documents:
            return documents[name]

        # single partial exact match
        candidates = [doc for doc_name, doc in documents.items()
                      if name in doc_name]
        if len(candidates) == 1:
            return candidates[0]

        # fuzzy match
        if allow_fuzzy:
            ratios = []
            for doc_name in documents:
                ratios.append((fuzz.ratio(name, doc_name), doc_name))
            ratios.sort(key=lambda x: x[0], reverse=True)  # sort by ratio

            if (ratios[0][0] > 75) and (ratios[0][0] - ratios[1][0] > 25):
                return documents[ratios[0][1]]

        # no match but we had multiple partial matches, warn the user
        if len(candidates) > 1:
            warnings.warn(f"Multiple documents with partial match found for "
                          f"name '{name}': "
                          f"{', '.join(f"'{c.name}'" for c in candidates)}. ")

        raise ValueError(f"Document '{name}' not found.")

    @staticmethod
    def list_events(season: int) -> list[str]:
        """List all events for a given season.

        More recent events are listed first.

        Args:
            season: The season year.

        Returns:
            A list of event names.
        """
        return list(load_decision_documents(season).keys())


class FileSystemDocumentBackend:
    """
    A file system based backend for storing and loading FIA documents.

    The backend creates a JSON index file in the base directory to keep track
    of the documents that have been loaded. The index file is used to check if
    a document has been updated on the server since the last time it was loaded.
    The backend supports comparison by ETag, Last-Modified header and content
    hash.

    In a future version, the backend will optionally support storing the
    documents in the file system as well. Currently, only the metadata is
    stored in the index file. TODO: implement

    The backend supports three comparison criteria: ETag and Last-Modified from
    the HTTP response headers as well as a hash of the actual PDF content. When
    multiple criteria are enabled, the document is considered updated if any of
    them indicate an update.
    Using the hash comparison is the most reliable way to check for updates,
    but it requires downloading the document content, which can be slow. ETag
    and Last-Modified headers are faster as they only require a HEAD request.

    Args:
        base_dir: The base directory in which to store the documents and the
            index file.
        compare_by_etag: Whether to compare documents by ETag header.
        compare_by_last_modified: Whether to compare documents by
            Last-Modified header.
        compare_by_hash: Whether to compare documents by content hash.
    """

    FILENAME_REGEX = re.compile(r"(.+?)(?:\((\d{10})\))?\.pdf")
    INDEX = "index.json"

    def __init__(
            self,
            base_dir: str,
            compare_by_etag: bool = True,
            compare_by_last_modified: bool = True,
            compare_by_hash: bool = False
    ):
        self.base_dir = base_dir

        self._compare_by_etag = compare_by_etag
        self._compare_by_last_modified = compare_by_last_modified
        self._compare_by_hash = compare_by_hash

        self._index_path = os.path.join(self.base_dir, self.INDEX)
        self._index = None

        self.load_index()

    def load_index(self):
        """Load the index file from disk if it exists or create an empty index.
        """
        if os.path.exists(self._index_path):
            try:
                with open(self._index_path, "r") as f:
                    self._index = json.load(f)
            except json.JSONDecodeError:
                raise ValueError("Index file exists but is not a valid JSON "
                                 "file.")
        else:
            self._index = {}

    def commit_index(self):
        """Write the index to disk."""
        with open(self._index_path, "w") as f:
            json.dump(self._index, f, indent=4)

    def get_from_index(self, document: Document) -> Document | None:
        """Given any document object, return the cached version of the document
        from the local index (e.g. for comparison of new and old).

        Args:
            document: The reference document

        Returns:
            The cached document version from the local index or None if it is
            not in the index.
        """
        data = self._index.get(self._get_document_key(document))
        if data:
            data["last_modified"] = dt_from_last_modified(
                data["last_modified"]
            )
            return Document(**data)
        return None

    @staticmethod
    def _get_document_key(document):
        # generate a unique index key for the document based on its metadata
        if document.is_decision_document:
            key = f"d#{document.season}-{document.event}-{document.name}"
        else:
            key = (f"t#{document.season}-{document.event}-{document.session}"
                   f"-{document.name}")
        return key

    def update_index(self, document: Document):
        """Update the index with the given document metadata.

        Note that this will not yet write the index to disk. Call
        ``commit_index`` to do so.

        Args:
            document (Document): The document that should be updated in the
                index or added if it is not already in the index.
        """
        self._index[self._get_document_key(document)] = document.to_json()

    def load_document(self, document: Document):
        """Load the given document from the server and update the index."""
        response = requests.get(get_url(document.path))

        document.etag = response.headers.get("ETag")
        document.last_modified = dt_from_last_modified(
            response.headers.get("Last-Modified")
        )
        document.hash = hashlib.md5(response.content).hexdigest()
        document.content = response.content
        document.was_updated = self.was_updated(document, response=response)

        self.update_index(document)
        self.commit_index()

    def was_updated(
            self,
            document: Document,
            *,
            response: requests.Response | None = None
    ) -> bool:
        """
        Check if the given document has been updated based on the comparison
        criteria. If multiple criteria are enabled, the document is considered
        updated if any of them indicate an update.

        Args:
            document (Document): The document to check for updates.
            response (requests.Response, optional): The response object of the
                document's content. If not provided, the content will be fetched
                from the URL. Defaults to None. This is useful if the content
                has already been fetched. You are responsible for ensuring that
                the response object is up-to-date, correct and contains the
                content of the document if hash comparison is enabled.

        Returns:
            bool: True if the document has been updated, False otherwise.
        """
        old_document = self.get_from_index(document)

        if not old_document:
            return True

        # If the document does not have the required comparison attribute,
        # consider it updated
        if self._compare_by_etag and not old_document.etag:
            return True
        if self._compare_by_last_modified and not old_document.last_modified:
            return True
        if self._compare_by_hash and not old_document.hash:
            return True

        # Only fetch the document content if necessary for hash comparison,
        # otherwise use a HEAD request to get the headers only
        if response is not None:
            pass  # already fetched
        elif self._compare_by_hash:
            response = requests.get(get_url(document.path))
        else:
            response = requests.head(get_url(document.path))

        was_updated = False

        # Compare ETag header
        if self._compare_by_etag:
            was_updated |= (response.headers["ETag"] != old_document.etag)

        # Compare Last-Modified header
        if self._compare_by_last_modified:
            last_modified = dt_from_last_modified(
                response.headers["Last-Modified"]
            )
            if not old_document.last_modified:
                was_updated = True
            else:
                was_updated |= (last_modified > old_document.last_modified)

        # Compare content hash
        if self._compare_by_hash:
            new_hash = hashlib.md5(response.content).hexdigest()
            was_updated |= (new_hash != old_document.hash)

        return was_updated
