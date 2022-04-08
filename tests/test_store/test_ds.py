import unittest as _unittest

from pyinstr_iakoster.store import (
    DataSpace, DataSpaceTemplate
)


class TestDataSpace(_unittest.TestCase):

    class DS0(DataSpace):

        _mul_rules = {0: ('a', 'b'),
                  'b0': ('b',),
                  'a': ('b',)}

        a = 1
        b = 'c'

    def test_attrs(self) -> None:
        self.assertEqual(1, self.DS0.a)

    def test_names(self) -> None:
        self.assertSetEqual({'a', 'b'}, self.DS0.vars())

    def test_attr(self) -> None:
        self.assertEqual('c', self.DS0.var('b'))

    def test_rule_one(self) -> None:
        self.assertTupleEqual(('c',), self.DS0.var('b0'))

    def test_rule_several(self) -> None:
        self.assertTupleEqual((1, 'c'), self.DS0.var(0))

    def test_rule_attr_exists(self) -> None:
        self.assertEqual(1, self.DS0.var('a'))

    def test_rule_noexists(self) -> None:
        with self.assertRaises(AttributeError):
            self.assertEqual(1, self.DS0.var('abc'))


class TestDataSpaceInit(_unittest.TestCase):

    def setUp(self) -> None:
        self.dsm = DataSpace()

    def test_init(self) -> None:
        self.dsm.b = 5
        self.assertEqual(self.dsm.b, 5)


class TestDataSpaceTemplate(_unittest.TestCase):

    st_data = dict(a=10, b=12, c=13, d='test')

    class DSTempTest(DataSpaceTemplate):

        _redirect_rules = {12: 'a', 'ph3': 'c'}
        _mul_rules = {0: ('a',),
                      1: ('d', 'c'),
                  'd': ('b',),
                  'e': ('a', 'b')}

        a = 10
        b = 12
        c: int
        d: str = 'test'

    def setUp(self) -> None:
        self.ds = self.DSTempTest(**self.st_data)

    def test_attrs(self) -> None:
        self.assertEqual(10, self.ds.a)

    def test_names(self) -> None:
        self.assertSetEqual({'a', 'b', 'c', 'd'}, self.ds.vars())

    def test_init_full(self) -> None:
        data = dict(a=2341, b=1241234, c=8, d='test2')
        ds = self.DSTempTest(**data)
        for attr in ds.vars():
            with self.subTest(attr=attr):
                self.assertEqual(data[attr], ds[attr])

    def test_init_partial(self) -> None:
        data = dict(a=10, b=12, c=124, d='test3')
        init_data = dict(c=124, d='test3')
        ds = self.DSTempTest(**init_data)
        for attr in ds.vars():
            with self.subTest(attr=attr):
                self.assertEqual(data[attr], ds[attr])

    def test_init_req(self) -> None:
        data = dict(a=10, b=12, c=124, d='test')
        init_data = dict(c=124)
        ds = self.DSTempTest(**init_data)
        for attr in ds.vars():
            with self.subTest(attr=attr):
                self.assertEqual(data[attr], ds[attr])

    def test_wrong_type(self) -> None:
        with self.assertRaises(TypeError) as exc:
            self.DSTempTest(c='test')
        self.assertEqual(
            'The annotation of \'c\' is different from '
            'the real type (exp/rec): '
            '<class \'int\'>/<class \'str\'>',
            exc.exception.args[0]
        )

    def test_not_all(self) -> None:
        with self.assertRaises(AttributeError) as exc:
            self.DSTempTest()
        self.assertEqual(
            'Attributes {\'c\'} is undefined',
            exc.exception.args[0]
        )

    def test_rule_attr_one(self) -> None:
        self.assertTupleEqual((10,), self.ds.var(0))

    def test_rule_item_several(self) -> None:
        self.assertTupleEqual(('test', 13), self.ds[1])

    def test_rule_attr_exists(self) -> None:
        self.assertEqual('test', self.ds.var('d'))

    def test_rule_direct(self) -> None:
        self.assertTupleEqual((10, 12), self.ds.e)

    def test_rule_noexists(self) -> None:
        with self.assertRaises(AttributeError):
            self.assertEqual(1, self.ds.abc)

    def test_redirect(self):
        self.assertEqual(13, self.ds.ph3)

    def test_redirect_attr(self):
        self.assertEqual(10, self.ds.var(12))

    def test_redirect_item(self):
        self.assertEqual(10, self.ds[12])
