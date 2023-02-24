import shutil
import unittest

from tests_deprecated.env_vars import TEST_DATA_DIR

from pyiak_instr_deprecation.rwfile import RWSQLite

SQLLITE_NAME = 'test_sqllite.db'
SQLLITE_PATH = TEST_DATA_DIR / SQLLITE_NAME


class TestRWSimpleSqlLite(unittest.TestCase):

    table = 'test_table'

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(TEST_DATA_DIR)

    def setUp(self) -> None:
        self.rws = RWSQLite(SQLLITE_PATH, timeout=0.5)

    def tearDown(self) -> None:
        self.rws.close()

    def test_0a_create_table_new(self):
        RWSQLite(TEST_DATA_DIR / 'test.database_aa-lol.db', timeout=0.5)
        self.assertTrue((TEST_DATA_DIR / 'test.database_aa-lol.db').exists())

    def test_a_create_table(self):
        self.rws.create_table(
            self.table,
            id='INTEGER PRIMARY KEY AUTOINCREMENT',
            name='TEXT',
            price='FLOAT')
        self.assertListEqual(
            ['id', 'name', 'price'],
            self.rws.table_columns(self.table)
        )

    def test_b_insert_into(self):
        self.rws.request(
            f"INSERT INTO {self.table} VALUES (?, ?, ?)",
            (0, 'first', 192.)
        )
        self.assertTupleEqual(
            (0, 'first', 192.),
            self.rws.hapi.execute(
                'SELECT * FROM test_table WHERE id=0;'
            ).fetchall()[0]
        )

    def test_c_insert_into_many(self):
        self.rws.request(
            f"INSERT INTO {self.table} VALUES (?, ?, ?)",
            [(1, 'second', 92.8), (2, 'third', 47.)]
        )
        self.assertListEqual(
            [(1, 'second', 92.8), (2, 'third', 47.0)],
            self.rws.hapi.execute(
                'SELECT * FROM test_table WHERE id IN (1, 2);'
            ).fetchall()
        )

    def test_d_select(self):
        self.rws.request(
            f"INSERT INTO {self.table} VALUES (?, ?, ?)",
            (4, 'fourth', 88)
        )
        result = self.rws.request(
            f'SELECT * FROM {self.table} WHERE id=4'
        ).fetchall()[0]
        self.assertTupleEqual(
            (4, 'fourth', 88.0),
            result
        )

    def test_e_insert_into_autoincrement(self):
        self.rws.request(
            f"INSERT INTO {self.table}(name, price) VALUES (?, ?)",
            ('five', 152.)
        )
        rows_count = self.rws.table_rows(self.table)
        result = self.rws.request(
            f'SELECT * FROM {self.table} WHERE id={rows_count}'
        ).fetchall()[0]
        self.assertTupleEqual(
            (rows_count, 'five', 152.),
            result
        )

    def test_w_tables(self):
        self.assertListEqual(
            ['test_table', 'sqlite_sequence'],
            self.rws.tables)

    def test_x_columns(self):
        self.assertDictEqual(
            {'test_table': ['id', 'name', 'price'], 'sqlite_sequence': ['name', 'seq']},
            self.rws.columns)

    def test_y_rows_count(self):
        self.assertDictEqual(
            {'test_table': 5, 'sqlite_sequence': 1},
            self.rws.rows)

    def test_z_delete_from(self):
        self.assertEqual(5, self.rws.table_rows(self.table))
        self.rws.request('DELETE FROM %s;' % self.table)
        self.assertEqual(0, self.rws.table_rows(self.table))

    def test_str_magic(self):
        self.assertEqual(
            r"RWSQLite('data_test\test_sqllite.db')", str(self.rws)
        )

