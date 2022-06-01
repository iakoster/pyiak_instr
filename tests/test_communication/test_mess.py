import unittest

from pyinstr_iakoster.communication import FieldSetter, Message


class TestFieldSetter(unittest.TestCase):

    def validate_setter(
            self,
            fs: FieldSetter,
            args: tuple,
            kwargs: dict,
            special: str = None
    ):
        self.assertTupleEqual(args, fs.args)
        self.assertDictEqual(kwargs, fs.kwargs)
        self.assertEqual(special, fs.special)

    def test_init(self):
        self.validate_setter(
            FieldSetter(1, 2, a=0, b=3),
            (1, 2),
            {"a": 0, "b": 3}
        )

    def test_base(self):
        self.validate_setter(
            FieldSetter.base(1, "i"),
            (1, "i"),
            {"content": b"", "info": None}
        )

    def test_single(self):
        self.validate_setter(
            FieldSetter.single("i"),
            ("i",),
            {"content": b"", "info": None},
            special="single"
        )

    def test_static(self):
        self.validate_setter(
            FieldSetter.static("i", b""),
            ("i", b""),
            {"info": None},
            special="static"
        )

    def test_address(self):
        self.validate_setter(
            FieldSetter.address("i"),
            ("i",),
            {"content": b"", "info": None}
        )

    def test_data(self):
        self.validate_setter(
            FieldSetter.data(3, "i"),
            (3, "i"),
            {"content": b"", "info": None}
        )

    def test_data_length(self):
        self.validate_setter(
            FieldSetter.data_length("i"),
            ("i",),
            {"additive": 0, "content": b"", "info": None, "units": 16}
        )

    def test_operation(self):
        self.validate_setter(
            FieldSetter.operation("i"),
            ("i",),
            {"content": b"", "desc_dict": None, "info": None}
        )
