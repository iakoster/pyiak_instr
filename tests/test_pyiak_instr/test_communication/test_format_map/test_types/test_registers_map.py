import unittest

import pandas as pd
from pandas.testing import assert_series_equal, assert_frame_equal

from src.pyiak_instr.core import Code
from src.pyiak_instr.exceptions import NotAmongTheOptions
from src.pyiak_instr.communication.message import (
    MessagePattern,
    MessageStructPattern,
    MessageFieldStructPattern,
)

from .....utils import validate_object
from .ti import TIRegister, TIRegistersMap


class TestRegisterABC(unittest.TestCase):

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
            wo_attrs=["series"]
        )

    def test_init_exc(self) -> None:
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
            "rw_type option not in {<Code.READ_ONLY: 1552>, "
            "<Code.WRITE_ONLY: 1553>, <Code.ANY: 5>}, got <Code.U8: 520>",
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
            f0=b"\x00\x14", f1=b"\x01", f2=b"\x00", f3=b""
        ).items():
            self.assertEqual(ref, msg.content(name))

    def test_read(self) -> None:
        with self.subTest(test="basic"):
            msg = self._instance().read(self._pattern())
            self.assertEqual(b"\x00\x14\x00\x2a", msg.content())
            for name, ref in dict(
                    f0=b"\x00\x14", f1=b"\x00", f2=b"\x2a", f3=b""
            ).items():
                self.assertEqual(ref, msg.content(name))

        with self.subTest(test="dynamic length is actual"):
            msg = self._instance().read(
                self._pattern(dlen_behaviour=Code.ACTUAL)
            )
            self.assertEqual(b"\x00\x14\x00\x00", msg.content())
            for name, ref in dict(
                    f0=b"\x00\x14", f1=b"\x00", f2=b"\x00", f3=b""
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
                    f2=b"\x01",
                    f3=b"\x00\x00\x00\x01",
            ).items():
                self.assertEqual(ref, msg.content(name))

    def test_from_series(self) -> None:
        ref = self._instance()
        self.assertEqual(ref, TIRegister.from_series(ref.series))

        self.assertEqual(
            TIRegister(
                pattern="pat",
                name="test",
                address=20,
                length=42,
            ),
            TIRegister.from_series(pd.Series(dict(
                pattern="pat",
                name="test",
                address=20,
                length=42,
                description=None
            ))),
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
                description="Short. Long."
            )),
        )

    @staticmethod
    def _pattern(
            dlen_behaviour: Code = Code.EXPECTED
    ) -> MessagePattern:
        return MessagePattern.basic().configure(
            s0=MessageStructPattern.basic().configure(
                f0=MessageFieldStructPattern.address(fmt=Code.U16),
                f1=MessageFieldStructPattern.operation(),
                f2=MessageFieldStructPattern.dynamic_length(
                    units=Code.WORDS, behaviour=dlen_behaviour
                ),
                f3=MessageFieldStructPattern.data(fmt=Code.U32),
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


class TestRegisterMapABC(unittest.TestCase):

    def test_init(self) -> None:
        validate_object(
            self,
            self._instance(),
            wo_attrs=["table"],
        )

    def test_init_exc(self) -> None:
        with self.assertRaises(ValueError) as exc:
            TIRegistersMap(
                pd.DataFrame(
                    columns=["name", "address", "length", "rw_type"],
                )
            )
        self.assertEqual(
            "missing columns in table: {'pattern'}", exc.exception.args[0]
        )

    def test_get_register(self) -> None:
        self.assertEqual(
            TIRegister(
                pattern="pat",
                name="test_0",
                address=42,
                length=20,
                rw_type=Code.ANY,
            ),
            self._instance().get_register("test_0")
        )

    def test_get_register_exc(self) -> None:
        instance = TIRegistersMap(
                pd.DataFrame(
                    columns=[
                        "pattern", "name", "address", "length", "rw_type"
                    ],
                    data=[
                        [None, "t", 0, 0, Code.ANY],
                        [None, "t", 0, 0, Code.ANY],
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
        obj = TIRegistersMap.from_registers(
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
                ],
                data=[
                    ["pat", "0", 0, 1, Code.ANY, ""],
                    ["pat", "1", 1, 1, 5, ""],
                ],
            ),
            obj.table,
        )

    @staticmethod
    def _instance() -> TIRegistersMap:
        return TIRegistersMap(
            pd.DataFrame(
                columns=["pattern", "name", "address", "length", "rw_type"],
                data=[
                    ["pat", "test_0", 42, 20, Code.ANY],
                    ["pat", "test_1", 43, 22, Code.READ_ONLY],
                    ["pat", "test_2", 44, 4, Code.WRITE_ONLY],
                ]
            )
        )