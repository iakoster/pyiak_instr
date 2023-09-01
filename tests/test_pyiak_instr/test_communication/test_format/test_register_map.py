import sqlite3
import unittest

import pandas as pd
from pandas.testing import assert_series_equal, assert_frame_equal

from src.pyiak_instr.core import Code
from src.pyiak_instr.exceptions import NotAmongTheOptions
from src.pyiak_instr.communication.format import RegisterMap
from src.pyiak_instr.testing import validate_object, compare_objects

from ...env import get_local_test_data_dir, remove_test_data_dir
from tests.pyiak_instr_ti.communication.message import (
    TIMessagePattern,
    TIStructPattern,
    TIFieldPattern,
)
from tests.pyiak_instr_ti.communication.format import TIRegister, TIRegisterMap


TEST_DATA_DIR = get_local_test_data_dir(__name__)


class TestRegister(unittest.TestCase):

    def test_init(self) -> None:
        validate_object(
            self,
            self._instance(),
            address=20,
            length=42,
            name="test",
            description="Short. Long.",
            pattern="pat",
            rw_type=Code.ANY,
            short_description="Short",
            message_kw="\dct()",
            struct_kw="\dct()",
            fields_kw="\dct()",
            wo_attrs=["series"]
        )

    def test_init_exc(self) -> None:
        with self.subTest(test="invalid rw_type"):
            with self.assertRaises(NotAmongTheOptions) as exc:
                TIRegister(
                    pattern="pat",
                    name="test",
                    address=20,
                    length=42,
                    description="Short. Long.",
                    rw_type=Code.U8,
                )
            self.assertEqual(
                "'rw_type' option <Code.U8: 520> not in {<Code.READ_ONLY: "
                "1552>, <Code.WRITE_ONLY: 1553>, <Code.ANY: 5>}",
                exc.exception.args[0],
            )

    def test_get(self) -> None:
        msg = self._instance().get(
            self._pattern(),
            operation=Code.WRITE,
            fields_data={"f1": Code.READ},
        )

        self.assertEqual(b"\x00\x14\x01\x00", msg.content())
        for name, ref in dict(
            f0=b"\x00\x14", f1=b"\x01", f3=b"\x00", f4=b""
        ).items():
            self.assertEqual(ref, msg.content(name))

    def test_get_additions(self) -> None:
        adds = TIRegister(
            pattern="",
            name="",
            address=0,
            length=1,
            message_kw="\dct(a,10)",
            struct_kw="\dct(b,20)",
            fields_kw="\dct(a,\dct(c,30))",
        ).get_additions("s0")

        self.assertDictEqual({"a": 10}, adds.current)
        self.assertDictEqual({"b": 20}, adds.lower("s0").current)
        self.assertDictEqual({"c": 30}, adds.lower("s0").lower("a").current)

    def test_read(self) -> None:
        with self.subTest(test="basic"):
            msg = self._instance().read(self._pattern())
            self.assertEqual(b"\x00\x14\x00\x2a", msg.content())
            for name, ref in dict(
                    f0=b"\x00\x14", f1=b"\x00", f3=b"\x2a", f4=b""
            ).items():
                self.assertEqual(ref, msg.content(name))

        with self.subTest(test="dynamic length is actual"):
            msg = self._instance().read(
                self._pattern(dlen_behaviour=Code.ACTUAL)
            )
            self.assertEqual(b"\x00\x14\x00\x00", msg.content())
            for name, ref in dict(
                    f0=b"\x00\x14", f1=b"\x00", f3=b"\x00", f4=b""
            ).items():
                self.assertEqual(ref, msg.content(name))

    def test_write(self) -> None:
        with self.subTest(test="basic"):
            msg = self._instance().write(self._pattern(), 1)
            self.assertEqual(
                b"\x00\x14\x01\x01\x00\x00\x00\x01", msg.content()
            )
            for name, ref in dict(
                    f0=b"\x00\x14",
                    f1=b"\x01",
                    f3=b"\x01",
                    f4=b"\x00\x00\x00\x01",
            ).items():
                self.assertEqual(ref, msg.content(name))

    def test_from_series(self) -> None:
        ref = self._instance()

        validate_object(
            self,
            TIRegister.from_series(pd.Series(dict(
                pattern="pat",
                name="test",
                address=20,
                length=42,
                description=None,
                message_kw="\dct()",
                struct_kw="\dct()",
                fields_kw="\dct()",
            ))),
            length=42,
            address=20,
            name="test",
            pattern="pat",
            description="",
            rw_type=Code.ANY,
            short_description="",
            wo_attrs=["series"],
            message_kw="\dct()",
            struct_kw="\dct()",
            fields_kw="\dct()",
        )

    def test_series(self) -> None:
        assert_series_equal(
            self._instance().series,
            pd.Series(dict(
                pattern="pat",
                name="test",
                address=20,
                length=42,
                rw_type=Code.ANY,
                description="Short. Long.",
                message_kw="\dct()",
                struct_kw="\dct()",
                fields_kw="\dct()",
            )),
        )

    @staticmethod
    def _pattern(
            dlen_behaviour: Code = Code.EXPECTED
    ) -> TIMessagePattern:
        return TIMessagePattern.basic().configure(
            s0=TIStructPattern.basic().configure(
                f0=TIFieldPattern.address(fmt=Code.U16),
                f1=TIFieldPattern.operation(),
                f2=TIFieldPattern.response(direction=Code.RX),
                f3=TIFieldPattern.dynamic_length(
                    units=Code.WORDS, behaviour=dlen_behaviour
                ),
                f4=TIFieldPattern.data(fmt=Code.U32),
            )
        )

    @staticmethod
    def _instance() -> TIRegister:
        return TIRegister(
                pattern="pat",
                name="test",
                address=20,
                length=42,
                description="Short. Long.",
            )


class TestRegisterMap(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        TEST_DATA_DIR.mkdir(parents=True, exist_ok=True)

    @classmethod
    def tearDownClass(cls) -> None:
        remove_test_data_dir()

    def test_init(self) -> None:
        validate_object(
            self,
            self._instance(),
            wo_attrs=["table"],
        )

    def test_init_exc(self) -> None:
        with self.subTest(test="without required columns"):
            with self.assertRaises(ValueError) as exc:
                TIRegisterMap(
                    pd.DataFrame(
                        columns=[
                            "name",
                            "address",
                            "length",
                            "rw_type",
                            "description",
                            "message_kw",
                            "struct_kw",
                            "fields_kw",
                        ],
                    )
                )
            self.assertEqual(
                "missing columns in table: {'pattern'}", exc.exception.args[0]
            )

        with self.subTest(test="withour _register_type"):
            with self.assertRaises(AttributeError) as exc:
                RegisterMap()
            self.assertEqual(
                "'RegisterMap' has no attribute '_register_type'",
                exc.exception.args[0],
            )

    def test_get_register(self) -> None:
        compare_objects(
            self,
            TIRegister(
                pattern="pat",
                name="test_0",
                address=42,
                length=20,
                rw_type=Code.ANY,
            ),
            self._instance().get_register("test_0"),
            wo_attrs=["additions", "series"],
        )

    def test_get_register_exc(self) -> None:
        instance = TIRegisterMap(
                pd.DataFrame(
                    columns=[
                        "pattern",
                        "name",
                        "address",
                        "length",
                        "rw_type",
                        "description",
                        "message_kw",
                        "struct_kw",
                        "fields_kw",
                    ],
                    data=[
                        [None, "t", 0, 0, Code.ANY, "", "\dct()", "\dct()", "\dct()"],
                        [None, "t", 0, 0, Code.ANY, "", "\dct()", "\dct()", "\dct()"],
                    ]
                )
            )

        with self.subTest(test="no one registers"):
            with self.assertRaises(ValueError) as exc:
                instance.get_register("reg")
            self.assertEqual(
                "register with name 'reg' not found",
                exc.exception.args[0],
            )

        with self.subTest(test="more than one register"):
            with self.assertRaises(ValueError) as exc:
                instance.get_register("t")
            self.assertEqual(
                "there is more than one register with the name 't'",
                exc.exception.args[0],
            )

    def test_from_registers(self) -> None:
        obj = TIRegisterMap.from_registers(
            TIRegister(pattern="pat", name="0", address=0, length=1),
            TIRegister(pattern="pat", name="1", address=1, length=1),
        )

        assert_frame_equal(
            pd.DataFrame(
                columns=[
                    "pattern",
                    "name",
                    "address",
                    "length",
                    "rw_type",
                    "description",
                    "message_kw",
                    "struct_kw",
                    "fields_kw",
                ],
                data=[
                    [
                        "pat",
                        "0",
                        0,
                        1,
                        Code.ANY,
                        "",
                        "\dct()",
                        "\dct()",
                        "\dct()",
                    ],
                    [
                        "pat",
                        "1",
                        1,
                        1,
                        5,
                        "",
                        "\dct()",
                        "\dct()",
                        "\dct()",
                    ],
                ],
            ),
            obj.table,
        )

    def test_write(self) -> None:
        path = TEST_DATA_DIR / "test_write.db"
        obj = self._instance()
        obj.write(path)

        with sqlite3.connect(path) as con:
            act = pd.read_sql("SELECT * FROM registers", con)

        assert_frame_equal(obj.table, act)

    def test_write_read(self) -> None:
        path = TEST_DATA_DIR / "test_write_read.db"
        ref = self._instance()
        ref.write(path)

        act = TIRegisterMap.read(path)
        assert_frame_equal(ref.table, act.table)

    @staticmethod
    def _instance() -> TIRegisterMap:
        return TIRegisterMap(
            pd.DataFrame(
                columns=[
                    "pattern",
                    "name",
                    "address",
                    "length",
                    "rw_type",
                    "description",
                    "message_kw",
                    "struct_kw",
                    "fields_kw",
                ],
                data=[
                    ["pat", "test_0", 42, 20, Code.ANY, "", "\dct()", "\dct()", "\dct()"],
                    ["pat", "test_1", 43, 22, Code.READ_ONLY, "", "\dct()", "\dct()", "\dct()"],
                    ["pat", "test_2", 44, 4, Code.WRITE_ONLY, "", "\dct()", "\dct()", "\dct()"],
                ]
            )
        )