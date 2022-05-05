import shutil
import unittest

import pymongo

from tests.env_vars import DATA_TEST_DIR

from pyinstr_iakoster.rwfile import RWNoSqlJsonCollection


NOSQL_NAME = 'test_nosql.json'
NOSQL_PATH = DATA_TEST_DIR / NOSQL_NAME


class TestRWNoSql(unittest.TestCase):

    table = 'test_table'

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(DATA_TEST_DIR)

    def test_a_create_db(self):
        path = DATA_TEST_DIR / 'test_db_2.json'
        RWNoSqlJsonCollection(path)
        self.assertTrue(path.exists())
