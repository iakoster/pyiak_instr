import re
import json
from pathlib import Path
from typing import Any, Iterable, MutableMapping, Mapping
from bson import json_util

from pymongo import database
import pymongo

from . import _operands as nso
from .._utils import *

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
    MASTER_FIELDS = {"_id", "type", "name", "desc", "coll_desc", "docs_count"}

    def __init__(
            self,
            filepath: Path | str,
            client: pymongo.MongoClient = None,
            database_name: str = "RWNoSqlJsonDatabase",
            host: str = "localhost",
            port: int = 27017,
            **client_kw: Any
    ):
        filepath = if_str2path(filepath)
        match_filename(self.FILENAME_PATTERN, filepath)
        create_dir_if_not_exists(filepath)
        self._filepath = filepath

        if client is None:
            self._client = pymongo.MongoClient(host, port, **client_kw)
        else:
            self._client = client
        self._db = self._client[database_name]
        self._coll = self._db[filepath.stem]

        if filepath.exists():
            self.load()
        self.validate_master()

    def insert_one(
            self, **fields: Any
    ) -> pymongo.collection.InsertOneResult:
        return self._coll.insert_one(fields)

    def insert_many(
            self, *docs: MutableMapping[str, Any]
    ) -> pymongo.collection.InsertManyResult:
        return self._coll.insert_many(docs)

    def insert(
            self,
            data: Iterable[
                      MutableMapping[str, Any]
                  ] | MutableMapping[str, Any]
    ) -> list[Any] | Any:
        """
        insert new data to the collection.

        Parameters
        ----------
        data Iterable of MutableMapping or MutableMapping
            new data for inserting.

        Returns
        -------
        list of Any or Any
            id or list of id if inserted data.

        Raises
        ------
        TypeError
            if data type is not Iterable or MutableMapping
        """
        if isinstance(data, Iterable):
            return self.insert_many(*tuple(data)).inserted_ids
        elif isinstance(data, MutableMapping):
            return self._coll.insert_one(**dict(data)).inserted_id
        else:
            raise TypeError("Invalid data type: %r" % data)

    def find_one(
            self, **fields
    ) -> Mapping[str, Any] | None:
        return self._coll.find_one(fields)

    def find(self, all_: bool = False, **fields: Any):
        if all_:
            return tuple()
        else:
            return self._coll.find_one(fields)

    def save(self) -> None:
        """Save collection to the JSON file by filepath"""
        cursor = self._coll.find({})
        with open(self._filepath, 'w') as file:
            json.dump(json.loads(json_util.dumps(cursor)), file)

    def load(self) -> None:
        """Load data from the JSON file by filepath and
        load it to the collection"""
        with open(self._filepath, 'r') as file:
            file_data = json.load(file)
            self._coll.insert_one(file_data)

    def validate_master(self) -> None:

        default = dict(
            _id=0,
            type="collection.master",
            name="master",
            desc=None,
            coll_desc=None,
            docs_count=self._coll.count_documents({})
        )
        assert set(default.keys()) == self.MASTER_FIELDS, \
            "Invalid master fields"

        master = self.find_one(_id=0)
        if master is None:
            self.insert_one(**default)
            return

        missing_keys = self.MASTER_FIELDS.difference(master.keys())
        extra_keys = set(master.keys()).difference(self.MASTER_FIELDS)
        if len(extra_keys) != 0:
            self._coll.update_one(
                {"_id": 0}, nso.UNSET(*tuple(extra_keys))
            )
        if len(missing_keys) != 0:
            self._coll.update_one(
                {"_id": 0}, nso.SET(**{f: default[f] for f in missing_keys})
            )
            print(default.fromkeys(missing_keys))

        master = self.find_one(_id=0)
        for field in ("type", "name", "docs_count"):
            if master[field] != default[field]:
                self._coll.update_one(
                    {"_id": 0}, nso.SET(**{field: default[field]})
                )

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
