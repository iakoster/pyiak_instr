import shutil
import unittest

import pymongo

from tests.env_vars import DATA_TEST_DIR

from pyinstr_iakoster.rwfile.nosql import RWNoSqlJsonCollection


NOSQL_NAME = 'test_nosql.json'
NOSQL_PATH = DATA_TEST_DIR / NOSQL_NAME


class TestRWNoSql(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.rwns = RWNoSqlJsonCollection(NOSQL_PATH)

    # @classmethod
    # def tearDownClass(cls) -> None:
    #     if DATA_TEST_DIR.exists():
    #         shutil.rmtree(DATA_TEST_DIR)

    def test_a_create_db(self):
        path = DATA_TEST_DIR / 'test_db_2.json'
        RWNoSqlJsonCollection(path)
        #self.assertTrue(path.exists())

    #def test_b_
