import unittest

import pandas as pd
from pandas.testing import assert_series_equal

from src.pyiak_instr.core import Code
from src.pyiak_instr.exceptions import NotAmongTheOptions
from src.pyiak_instr.communication.message import (
    MessagePattern,
    MessageStructPattern,
    MessageFieldStructPattern,
)

from .....utils import validate_object
from .ti import TIRegisterStruct


class TestRegisterStructABC(unittest.TestCase):

    def test_init(self) -> None:
        validate_object(
            self,
            self._instance(),
            address=20,
            length=42,
            name="test",
            description="Short. Long.",
            pattern=None,
            rw_type=Code.ANY,
            short_description="Short",
            wo_attrs=["series"]
        )

    def test_init_exc(self) -> None:
        with self.assertRaises(NotAmongTheOptions) as exc:
            TIRegisterStruct(
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
        ...

    def test_from_series(self) -> None:
        ref = self._instance()
        self.assertEqual(ref, TIRegisterStruct.from_series(ref.series))

        self.assertEqual(
            TIRegisterStruct(
                name="test",
                address=20,
                length=42,
            ),
            TIRegisterStruct.from_series(pd.Series(dict(
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
                name="test",
                address=20,
                length=42,
                rw_type=Code.ANY,
                description="Short. Long."
            )),
        )

    @staticmethod
    def _instance(pattern: MessagePattern | None = None) -> TIRegisterStruct:
        return TIRegisterStruct(
                name="test",
                address=20,
                length=42,
                description="Short. Long.",
                pattern=pattern,
            )
