from __future__ import annotations

import unittest
import shutil
from pathlib import Path
from configparser import ConfigParser
from dataclasses import dataclass, field as _field, InitVar
from typing import Any, ClassVar, Iterable, TypeAlias

import numpy as np
import numpy.typing as npt
from numpy.testing import assert_array_equal

from src.pyiak_instr.core import Code
from src.pyiak_instr.exceptions import NotConfiguredYet
from src.pyiak_instr.rwfile import RWConfig
from src.pyiak_instr.types import Encoder
from src.pyiak_instr.types.store.bin import (
    STRUCT_DATACLASS,
    BytesFieldStructABC,
    BytesFieldStructPatternABC,
    BytesStorageABC,
    BytesStoragePatternABC,
    BytesStorageStructABC,
    BytesStorageStructPatternABC,
)

from .....utils import validate_object, compare_objects
from ....env import TEST_DATA_DIR


class TIEncoder(Encoder):

    def __init__(self, fmt: Code = Code.U8, order: Code = Code.BIG_ENDIAN):
        if fmt not in {
            Code.U8, Code.U16, Code.U24, Code.U32, Code.U40
        }:
            raise ValueError("invalid fmt")
        if order is not Code.BIG_ENDIAN:
            raise ValueError("invalid order")
        self.fmt, self.order = fmt, order

    def decode(self, value: bytes) -> npt.NDArray[np.int_ | np.float_]:
        return np.frombuffer(
            value, np.uint8 if self.fmt is Code.U8 else np.uint16
        )

    def encode(self, value: int | float | Iterable[int | float]) -> bytes:
        if isinstance(value, bytes):
            return value
        return np.array(value).astype(
            np.uint8 if self.fmt is Code.U8 else np.uint16
        ).tobytes()

    @property
    def value_size(self) -> int:
        return (
            Code.U8, Code.U16, Code.U24, Code.U32, Code.U40
        ).index(self.fmt) + 1


@STRUCT_DATACLASS
class TIFieldStruct(BytesFieldStructABC):

    encoder: InitVar[type[TIEncoder]] = TIEncoder


@STRUCT_DATACLASS
class TIStorageStruct(BytesStorageStructABC[TIFieldStruct]):
    ...


class TIStorage(
    BytesStorageABC[TIFieldStruct, TIStorageStruct, "TIStoragePattern"]
):
    _storage_struct_type = TIStorageStruct


class TIFieldStructPattern(BytesFieldStructPatternABC[TIFieldStruct]):
    _options = {"basic": TIFieldStruct}


class TIStorageStructPattern(
    BytesStorageStructPatternABC[TIStorageStruct, TIFieldStructPattern]
):
    _sub_p_type = TIFieldStructPattern
    _options = {"basic": TIStorageStruct}


class TIStoragePattern(
    BytesStoragePatternABC[TIStorage, TIStorageStructPattern]
):
    _rwdata = RWConfig
    _sub_p_type = TIStorageStructPattern
    _options = {"basic": TIStorage}
