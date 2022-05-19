import shutil
import unittest

from tests.env_vars import DATA_TEST_DIR

from pyinstr_iakoster.rwfile import RWNoSqlJsonDatabase, Query


NOSQL_NAME = 'test_nosql.json'
NOSQL_PATH = DATA_TEST_DIR / NOSQL_NAME


class TestRWNoSql(unittest.TestCase):

    rwns: RWNoSqlJsonDatabase

    @classmethod
    def setUpClass(cls) -> None:
        cls.rwns = RWNoSqlJsonDatabase(NOSQL_PATH)

    @classmethod
    def tearDownClass(cls) -> None:
        if NOSQL_PATH.exists():
            cls.rwns.close()
            shutil.rmtree(DATA_TEST_DIR)

    def test_aa_create(self):
        self.assertEqual(self.rwns.count(Query()["name"].exists()), 0)

    def test_ab_insert(self):
        self.assertEqual(
            self.rwns.rwf_insert(name="test_ab_1", value=1000, type="a"), 1
        )
        self.assertEqual(
            self.rwns.rwf_insert(name="test_ab_2", value=-100, type="b"), 2
        )
        self.assertEqual(self.rwns.count(Query()["name"].exists()), 2)

    def test_ac_update(self):
        self.assertEqual(self.rwns.get(doc_id=1)["value"], 1000)
        self.rwns.rwf_insert(name="test_ab_1", value=0, type="c")
        self.assertListEqual(
            self.rwns.rwf_update(
                Query()["name"] == "test_ab_1", value=250, type="a"
            ),
            [1, 3]
        )
        self.assertEqual(self.rwns.get(doc_id=1)["value"], 250)
        self.assertEqual(self.rwns.get(doc_id=3)["type"], "a")
        self.assertEqual(self.rwns.get(doc_id=2)["value"], -100)
        self.assertListEqual(
            self.rwns.rwf_update(doc_ids=2, value=-120), [2]
        )
        self.assertEqual(self.rwns.get(doc_id=2)["value"], -120)

    def test_ad_upsert(self):
        self.assertListEqual(
            self.rwns.rwf_upsert(
                Query()["name"] == "test_ab_2", name="test_ab_2", value=0
            ), [2]
        )
        self.assertListEqual(
            self.rwns.rwf_upsert(
                Query()["name"] == "test_ad_upsert",
                name="test_ad_upsert",
                value=0
            ), [4]
        )

    def test_ae_table(self):
        self.rwns["test_table"].rwf_insert(name="test", value=10)
        self.assertIn("test_table", self.rwns.tables())
