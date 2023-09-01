import unittest

from src.pyiak_instr.rwfile import RWSQLite
from src.pyiak_instr.testing import validate_object


from ..env import get_local_test_data_dir, remove_test_data_dir

TEST_DATA_DIR = get_local_test_data_dir(__name__)


class TestRWSQLite(unittest.TestCase):

    DATABASE = "test.db"
    TEST_FILE = TEST_DATA_DIR / DATABASE

    @classmethod
    def tearDownClass(cls) -> None:
        remove_test_data_dir()

    def test_init(self) -> None:
        with RWSQLite(TEST_DATA_DIR / "test_2.db") as rws:
            validate_object(
                self,
                rws,
                columns={},
                filepath=TEST_DATA_DIR / "test_2.db",
                rows_counts={},
                tables=[],
                wo_attrs=["api", "connection"],
            )
            self.assertTrue((TEST_DATA_DIR / "test_2.db").exists())

    def test_request(self) -> None:
        with RWSQLite(self.TEST_FILE) as rws:
            rws.request(
                "CREATE TABLE "
                "test_table ("
                "id INTEGER PRIMARY KEY AUTOINCREMENT, "
                "name TEXT, "
                "price FLOAT"
                ");"
            )
            self.assertDictEqual(
                dict(
                    sqlite_sequence=['name', 'seq'],
                    test_table=['id', 'name', 'price'],
                ),
                rws.columns,
            )

            with self.subTest(test="insert"):
                rws.request(
                    "INSERT INTO test_table VALUES (?, ?, ?)",
                    (0, "first", 192.),
                )
                self.assertTupleEqual(
                    (0, "first", 192.),
                    rws.api.execute(
                        "SELECT * FROM test_table WHERE id=0;"
                    ).fetchall()[0]
                )

            with self.subTest(test="insert many"):
                rws.request(
                    "INSERT INTO test_table VALUES (?, ?, ?)",
                    [(1, "second", 92.8), (2, "third", 47.)]
                )
                self.assertListEqual(
                    [(1, "second", 92.8), (2, "third", 47.0)],
                    rws.api.execute(
                        "SELECT * FROM test_table WHERE id IN (1, 2);"
                    ).fetchall()
                )

            with self.subTest(test="validate"):
                validate_object(
                    self,
                    rws,
                    columns={
                        "sqlite_sequence": ["name", 'seq'],
                        "test_table": ["id", "name", "price"],
                    },
                    rows_counts={"sqlite_sequence": 1, "test_table": 3},
                    tables=["test_table", "sqlite_sequence"],
                    wo_attrs=["api", "connection", "filepath"],
                )
