import shutil
import unittest

from tests.env_vars import DATA_TEST_DIR

from pyinstr_iakoster.rwfile import RWSQLite3Simple
from pyinstr_iakoster.rwfile import FilepathPatternError

SQLLITE_NAME = 'test_sqllite.db'
SQLLITE_PATH = DATA_TEST_DIR / SQLLITE_NAME


class TestRWSimpleSqlLite(unittest.TestCase):

    table = 'test_table'

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(DATA_TEST_DIR)

    def setUp(self) -> None:
        self.rws = RWSQLite3Simple(SQLLITE_PATH, timeout=0.5)

    def tearDown(self) -> None:
        self.rws.close()

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
        self.rws.insert_into(
            into=self.table,
            values=(0, 'first', 192.))
        self.assertTupleEqual(
            (0, 'first', 192.),
            self.rws.cursor.execute('SELECT * FROM test_table WHERE id=0;').fetchall()[0]
        )

    def test_c_insert_into_many(self):
        self.rws.insert_into(
            into=self.table, values=[
                (1, 'second', 92.8),
                (2, 'third', 47.)
            ])
        self.assertListEqual(
            [(1, 'second', 92.8), (2, 'third', 47.0)],
            self.rws.cursor.execute('SELECT * FROM test_table WHERE id IN (1, 2);').fetchall()
        )

    def test_d_select(self):
        self.rws.insert_into(into=self.table, values=(4, 'fourth', 88))
        result = self.rws.select(from_=self.table, fetch='all', where='id=4')[0]
        self.assertTupleEqual(
            (4, 'fourth', 88.0),
            result
        )

    def test_e_insert_into_autoincrement(self):
        self.rws.insert_into(
            into=self.table,
            values=('five', 152.),
            columns=['name', 'price'])
        rows_count = self.rws.table_rows_count(self.table)
        result = self.rws.select(
            from_=self.table, fetch='all', where=f'id={rows_count}')[0]
        self.assertTupleEqual(
            (rows_count, 'five', 152.),
            result
        )

    def test_f_tables(self):
        self.assertListEqual(
            ['test_table', 'sqlite_sequence'],
            self.rws.tables)

    def test_g_columns(self):
        self.assertDictEqual(
            {'test_table': ['id', 'name', 'price'], 'sqlite_sequence': ['name', 'seq']},
            self.rws.columns)

    def test_h_rows_count(self):
        self.assertDictEqual(
            {'test_table': 5, 'sqlite_sequence': 1},
            self.rws.rows_count)

    def test_z_delete_from(self):
        self.assertEqual(5, self.rws.table_rows_count(self.table))
        self.rws.delete_from(from_=self.table)
        self.assertEqual(0, self.rws.table_rows_count(self.table))

