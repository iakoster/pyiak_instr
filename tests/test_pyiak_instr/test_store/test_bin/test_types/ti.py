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
from src.pyiak_instr.encoders import BytesEncoder
from src.pyiak_instr.encoders.types import Encoder
from src.pyiak_instr.store.bin.types import (
    STRUCT_DATACLASS,
    BytesFieldStructABC,
    BytesFieldStructPatternABC,
    BytesStorageABC,
    BytesStoragePatternABC,
    BytesStorageStructABC,
    BytesStorageStructPatternABC,
    ContinuousBytesStorageStructPatternABC,
)

from .....utils import validate_object, compare_objects
from ....env import TEST_DATA_DIR


@STRUCT_DATACLASS
class TIFieldStruct(BytesFieldStructABC):

    encoder: InitVar[type[BytesEncoder]] = BytesEncoder


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


class TIContinuousStorageStructPattern(
    ContinuousBytesStorageStructPatternABC[TIStorageStruct, TIFieldStructPattern]
):
    _sub_p_type = TIFieldStructPattern
    _options = {"basic": TIStorageStruct}


class TIStoragePattern(
    BytesStoragePatternABC[TIStorage, TIStorageStructPattern]
):
    _rwdata = RWConfig
    _sub_p_type = TIStorageStructPattern
    _options = {"basic": TIStorage}
