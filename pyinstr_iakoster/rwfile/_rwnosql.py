import re
import json
from pathlib import Path
from typing import Any, Iterable, MutableMapping
from bson import json_util

from pymongo import database
import pymongo

from ._utils import *

__all__ = ['RWNoSqlJsonCollection']


class RWNoSqlJsonCollection(object):
    """
    Class for work with JSON NoSql database
    collection by MongoDB

    Parameters
    ----------
    filepath: Path or path-like str
        path to the database.
    **client_kw: Any
        kwargs which will be passed when
        creating the MogoClient
    """

    FILENAME_PATTERN = re.compile('\S+.json$')

    def __init__(self, filepath: Path | str, **client_kw: Any):
        filepath = if_str2path(filepath)
        match_filename(self.FILENAME_PATTERN, filepath)
        create_dir_if_not_exists(filepath)

        self._filepath = filepath

        self._client = pymongo.MongoClient(**client_kw)
        self._db = self._client['database']
        self._coll = self._db['collection']
        if not filepath.exists():
            self.save()
        else:
            self.load()

    def save(self) -> None:
        """Save collection to the JSON file by filepath"""
        cursor = self._coll.find({})
        with open(self._filepath, 'w') as file:
            if cursor.retrieved != 0:
                json.dump(json.loads(json_util.dumps(cursor)), file)

    def load(self) -> None:
        """Load data from the JSON file by filepath and
        load it to the collection"""
        with open(self._filepath, 'r') as file:
            if len(file.read()) != 0:
                file_data = json.load(file)
                self.insert(file_data)

    def insert(
            self,
            data: Iterable[
                      MutableMapping[str, Any]
                  ] | MutableMapping[str, Any]
    ) -> list[Any] | Any:
        """
        insert new data to the collection

        Parameters
        ----------
        data Iterable of MutableMapping or MutableMapping
            new data for inserting

        Returns
        -------
        list of Any or Any
            id or list of id if inserted data
        """
        if isinstance(data, list):
            return self._coll.insert_many(data).inserted_ids
        else:
            return self._coll.insert_one(data).inserted_id

    def close(self) -> None:
        """Close client connection"""
        self._client.close()

    @property
    def client(self) -> pymongo.MongoClient:
        """
        Returns
        -------
        pymongo.MongoClient
            MongoDB client
        """
        return self._client

    @property
    def database(self) -> database.Database:
        """
        Returns
        -------
        database.Database
            MongoDB database
        """
        return self._db

    @property
    def collection(self) -> pymongo.collection.Collection:
        """
        Returns
        -------
        pymongo.collection.Collection
            MongoDB collection
        """
        return self._coll

    @property
    def filepath(self) -> Path:
        """
        Returns
        -------
        Path
            path to the json database
        """
        return self._filepath

    def __del__(self):
        self.close()
